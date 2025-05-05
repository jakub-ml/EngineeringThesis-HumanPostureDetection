import sys
sys.path.append("./mmpose") # utilities in child directory
import cv2 
import numpy as np 
import utils.utilsDataman
# import pandas as pd
import os 
import glob 
import pickle
import json
import subprocess
import urllib.request
import shutil
# import requests
import ffmpeg
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import gaussian, sosfiltfilt, butter, find_peaks
from scipy.interpolate import pchip_interpolate
# from scipy.spatial.transform import Rotation 
import scipy.linalg
from itertools import combinations
import copy
from warnings import warn
import yaml
import math
from scipy.special import cbrt
import time
import itertools

def rotateIntrinsics(CamParams,videoPath):
    rotation = getVideoRotation(videoPath)
    
    # upright is 90, which is how intrinsics recorded so no rotation needed
    if rotation !=90:
        originalCamParams = copy.deepcopy(CamParams)
        fx = originalCamParams['intrinsicMat'][0,0]
        fy = originalCamParams['intrinsicMat'][1,1]
        px = originalCamParams['intrinsicMat'][0,2]
        py = originalCamParams['intrinsicMat'][1,2]
        lx = originalCamParams['imageSize'][1]
        ly = originalCamParams['imageSize'][0]
           
        if rotation == 0: # leaning left
            # Flip image size
            CamParams['imageSize'] = np.flipud(CamParams['imageSize'])
            # Flip focal lengths
            CamParams['intrinsicMat'][0,0] = fy
            CamParams['intrinsicMat'][1,1] = fx
            # Change principle point - from upper left
            CamParams['intrinsicMat'][0,2] = py
            CamParams['intrinsicMat'][1,2] = lx-px   
        elif rotation == 180: # leaning right
            # Flip image size
            CamParams['imageSize'] = np.flipud(CamParams['imageSize'])
            # Flip focal lengths
            CamParams['intrinsicMat'][0,0] = fy
            CamParams['intrinsicMat'][1,1] = fx
            # Change principle point - from upper left
            CamParams['intrinsicMat'][0,2] = ly-py
            CamParams['intrinsicMat'][1,2] = px
        elif rotation == 270: # upside down
            # Change principle point - from upper left
            CamParams['intrinsicMat'][0,2] = lx-px
            CamParams['intrinsicMat'][1,2] = ly-py
            
    return CamParams


def calcExtrinsicsFromVideo(videoPath, CamParams, CheckerBoardParams,
                            visualize=False, imageUpsampleFactor=2,
                            useSecondExtrinsicsSolution=False):  
    # Get video parameters.
    vidLength = getVideoLength(videoPath)
    videoDir, videoName = os.path.split(videoPath)    
    # Pick end of video as only sample point. For some reason, won't output
    # video with t close to vidLength, so we count down til it does.
    tSampPts = [np.round(vidLength-0.3, decimals=1)]    
    upsampleIters = 0
    for iTime,t in enumerate(tSampPts):
        # Pop an image.
        imagePath = os.path.join(videoDir, 'extrinsicImage0.png')
        if os.path.exists(imagePath):
            os.remove(imagePath)
        while not os.path.exists(imagePath) and t>=0:
            video2Images(videoPath, nImages=1, tSingleImage=t, filePrefix='extrinsicImage', skipIfRun=False)
            t -= 0.2
        # Default to beginning if can't find a keyframe.
        if not os.path.exists(imagePath):
            video2Images(videoPath, nImages=1, tSingleImage=0.01, filePrefix='extrinsicImage', skipIfRun=False)
        # Throw error if it can't find a keyframe.
        if not os.path.exists(imagePath):
            exception = 'No calibration image could be extracted for at least one camera. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration and https://www.opencap.ai/troubleshooting for potential causes for a failed calibration.'
            raise Exception(exception, exception)
        # Try to find the checkerboard; return None if you can't find it.    
        
        CamParamsTemp = calcExtrinsics(
            os.path.join(videoDir, 'extrinsicImage0.png'),
            CamParams, CheckerBoardParams, visualize=visualize, 
            imageUpsampleFactor=imageUpsampleFactor,
            useSecondExtrinsicsSolution=useSecondExtrinsicsSolution)
        while iTime == 0 and CamParamsTemp is None and upsampleIters < 3:
            if imageUpsampleFactor > 1: 
                imageUpsampleFactor = 1
            elif imageUpsampleFactor == 1:
                imageUpsampleFactor = .5
            elif imageUpsampleFactor < 1:
                imageUpsampleFactor = 1
            CamParamsTemp = calcExtrinsics(
                os.path.join(videoDir, 'extrinsicImage0.png'),
                CamParams, CheckerBoardParams, visualize=visualize, 
                imageUpsampleFactor=imageUpsampleFactor,
                useSecondExtrinsicsSolution=useSecondExtrinsicsSolution)
            upsampleIters += 1
        if CamParamsTemp is not None:
            # If checkerboard was found, exit.
            CamParams = CamParamsTemp.copy()
            return CamParams

    
    return None


def calcExtrinsics(imageFileName, CameraParams, CheckerBoardParams,
                   imageScaleFactor=1,visualize=False,
                   imageUpsampleFactor=1,useSecondExtrinsicsSolution=False):
    # Camera parameters is a dictionary with intrinsics
    
    # stop the iteration when specified 
    # accuracy, epsilon, is reached or 
    # specified number of iterations are completed. 
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
      
    # Vector for 3D points 
    threedpoints = [] 
      
    # Vector for 2D points 
    twodpoints = [] 
    
    #  3D points real world coordinates. Assuming z=0
    objectp3d = generate3Dgrid(CheckerBoardParams)
    
    # Load and resize image - remember calibration image res needs to be same as all processing
    image = cv2.imread(imageFileName)
    if imageScaleFactor != 1:
        dim = (int(imageScaleFactor*image.shape[1]),int(imageScaleFactor*image.shape[0]))
        image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
        
    if imageUpsampleFactor != 1:
        dim = (int(imageUpsampleFactor*image.shape[1]),int(imageUpsampleFactor*image.shape[0]))
        imageUpsampled = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    else:
        imageUpsampled = image

    
    # Find the chess board corners 
    # If desired number of corners are 
    # found in the image then ret = true 
    
    #TODO need to add a timeout to the findChessboardCorners function
    grayColor = cv2.cvtColor(imageUpsampled, cv2.COLOR_BGR2GRAY)
    
    ## Contrast TESTING - openCV does thresholding already, but this may be a bit helpful for bumping contrast
    # grayColor = grayColor.astype('float64')
    # cv2.imshow('Grayscale', grayColor.astype('uint8'))
    # savePath = os.path.join(os.path.dirname(imageFileName),'extrinsicGray.jpg')
    # cv2.imwrite(savePath,grayColor)
    
    # grayContrast = np.power(grayColor,2)
    # grayContrast = grayContrast/(np.max(grayContrast)/255)    
    # # plt.figure()
    # # plt.imshow(grayContrast, cmap='gray')
    
    # # cv2.imshow('ContrastEnhanced', grayContrast.astype('uint8'))
    # savePath = os.path.join(os.path.dirname(imageFileName),'extrinsicGrayContrastEnhanced.jpg')
    # cv2.imwrite(savePath,grayContrast)
      
    # grayContrast = grayContrast.astype('uint8')
    # grayColor = grayColor.astype('uint8')

    ## End contrast Testing
    
    ## Testing settings - slow and don't help 
    # ret, corners = cv2.findChessboardCorners( 
    #                 grayContrast, CheckerBoardParams['dimensions'],  
    #                 cv2.CALIB_CB_ADAPTIVE_THRESH  
    #                 + cv2.CALIB_CB_FAST_CHECK + 
    #                 cv2.CALIB_CB_NORMALIZE_IMAGE) 
    
    # Note I tried findChessboardCornersSB here, but it didn't find chessboard as reliably
    ret, corners = cv2.findChessboardCorners( 
                grayColor, CheckerBoardParams['dimensions'],  
                cv2.CALIB_CB_ADAPTIVE_THRESH) 

    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        # 3D points real world coordinates       
        threedpoints.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
        corners2 = cv2.cornerSubPix( 
            grayColor, corners, (11, 11), (-1, -1), criteria) / imageUpsampleFactor
  
        twodpoints.append(corners2) 
  
        # For testing: Draw and display the corners 
        # image = cv2.drawChessboardCorners(image,  
        #                                  CheckerBoardParams['dimensions'],  
        #                                   corners2, ret) 
        # Draw small dots instead
        # Choose dot size based on size of squares in pixels
        circleSize = 1
        squareSize = np.linalg.norm((corners2[1,0,:] - corners2[0,0,:]).squeeze())
        if squareSize >12:
            circleSize = 2

        for iPoint in range(corners2.shape[0]):
            thisPt = corners2[iPoint,:,:].squeeze()
            cv2.circle(image, tuple(thisPt.astype(int)), circleSize, (255,255,0), 2) 
        
        #cv2.imshow('img', image) 
        #cv2.waitKey(0) 
  
        #cv2.destroyAllWindows()
    if ret == False:
        print('No checkerboard detected. Will skip cam in triangulation.')
        return None
        
        
    # Find position and rotation of camera in board frame.
    # ret, rvec, tvec = cv2.solvePnP(objectp3d, corners2,
    #                                CameraParams['intrinsicMat'], 
    #                                CameraParams['distortion'])
    
    # This function gives two possible solutions.
    # It helps with the ambiguous cases with small checkerboards (appears like
    # left handed coord system). Unfortunately, there isn't a clear way to 
    # choose the correct solution. It is the nature of the solvePnP problem 
    # with a bit of 2D point noise.
    rets, rvecs, tvecs, reprojError = cv2.solvePnPGeneric(
        objectp3d, corners2, CameraParams['intrinsicMat'], 
        CameraParams['distortion'], flags=cv2.SOLVEPNP_IPPE)
    rvec = rvecs[1]
    tvec = tvecs[1]
   
    if rets < 1 or np.max(rvec) == 0 or np.max(tvec) == 0:
        print('solvePnPGeneric failed. Use SolvePnPRansac')
        # Note: can input extrinsics guess if we generally know where they are.
        # Add to lists to look like solvePnPRansac results
        rvecs = []
        tvecs = []
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectp3d, corners2, CameraParams['intrinsicMat'],
            CameraParams['distortion'])
        if ret is True:
            rets = 1
            rvecs.append(rvec)
            tvecs.append(tvec)
        else:
            print('Extrinsic calculation failed. Will skip cam in triangulation.')
            return None
    
    # Select which extrinsics solution to use
    extrinsicsSolutionToUse = 0
    if useSecondExtrinsicsSolution:
        extrinsicsSolutionToUse = 1
        
    topLevelExtrinsicImageFolder = os.path.abspath(
        os.path.join(os.path.dirname(imageFileName),
                     '../../../../Data/CalibrationImages'))
    if not os.path.exists(topLevelExtrinsicImageFolder):
        os.makedirs(topLevelExtrinsicImageFolder,exist_ok=True)
        
    for iRet,rvec,tvec in zip(range(rets),rvecs,tvecs):
        theseCameraParams = copy.deepcopy(CameraParams)
        # Show reprojections
        img_points, _ = cv2.projectPoints(objectp3d, rvec, tvec, 
                                          CameraParams['intrinsicMat'],  
                                          CameraParams['distortion'])
    
        # Plot reprojected points
        # for c in img_points.squeeze():
        #     cv2.circle(image, tuple(c.astype(int)), 2, (0, 255, 0), 2)
        
        # Show object coordinate system
        imageCopy = copy.deepcopy(image)
        imageWithFrame = cv2.drawFrameAxes(
            imageCopy, CameraParams['intrinsicMat'], 
            CameraParams['distortion'], rvec, tvec, 200, 4)
        
        # Create zoomed version.
        ht = image.shape[0]
        wd = image.shape[1]
        bufferVal = 0.05 * np.mean([ht,wd])
        topEdge = int(np.max([np.squeeze(np.min(img_points,axis=0))[1]-bufferVal,0]))
        leftEdge = int(np.max([np.squeeze(np.min(img_points,axis=0))[0]-bufferVal,0]))
        bottomEdge = int(np.min([np.squeeze(np.max(img_points,axis=0))[1]+bufferVal,ht]))
        rightEdge = int(np.min([np.squeeze(np.max(img_points,axis=0))[0]+bufferVal,wd]))
        
        # imageCopy2 = copy.deepcopy(imageWithFrame)
        imageCropped = imageCopy[topEdge:bottomEdge,leftEdge:rightEdge,:]
                
        
        # Save extrinsics picture with axis
        imageSize = np.shape(image)[0:2]
        #findAspectRatio
        ar = imageSize[1]/imageSize[0]
        # cv2.namedWindow("axis", cv2.WINDOW_NORMAL) 
        cv2.resize(imageWithFrame,(600,int(np.round(600*ar))))
     
        # save crop image to local camera file
        savePath2 = os.path.join(os.path.dirname(imageFileName), 
                                'extrinsicCalib_soln{}.jpg'.format(iRet))
        cv2.imwrite(savePath2,imageCropped)
          
        if visualize:   
            print('Close image window to continue')
            cv2.imshow('axis', image)
            cv2.waitKey()
            
            cv2.destroyAllWindows()
        
        R_worldFromCamera = cv2.Rodrigues(rvec)[0]
        
        theseCameraParams['rotation'] = R_worldFromCamera
        theseCameraParams['translation'] = tvec
        theseCameraParams['rotation_EulerAngles'] = rvec
        
        # save extrinsics parameters to video folder
        # will save the selected parameters in Camera folder in main
        saveExtPath = os.path.join(
            os.path.dirname(imageFileName),
            'cameraIntrinsicsExtrinsics_soln{}.pickle'.format(iRet))
        saveCameraParameters(saveExtPath,theseCameraParams)
        
        # save images to top level folder and return correct extrinsics
        camName = os.path.split(os.path.abspath(
                  os.path.join(os.path.dirname(imageFileName), '../../')))[1] 
            
        if iRet == extrinsicsSolutionToUse:
            fullCamName = camName 
            CameraParamsToUse = copy.deepcopy(theseCameraParams)
        else:
            fullCamName = 'altSoln_{}'.format(camName)
        savePath = os.path.join(topLevelExtrinsicImageFolder, 'extrinsicCalib_' 
                                + fullCamName + '.jpg')
        cv2.imwrite(savePath,imageCropped)   
            
    return CameraParamsToUse


def saveCameraParameters(filename,CameraParams):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename),exist_ok=True)
    
    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()
    
    return True

def generate3Dgrid(CheckerBoardParams):
    #  3D points real world coordinates. Assuming z=0
    objectp3d = np.zeros((1, CheckerBoardParams['dimensions'][0]  
                          * CheckerBoardParams['dimensions'][1],  
                          3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CheckerBoardParams['dimensions'][0], 
                                    0:CheckerBoardParams['dimensions'][1]].T.reshape(-1, 2) 
    
    objectp3d = objectp3d * CheckerBoardParams['squareSize'] 
    
    return objectp3d


def video2Images(videoPath, nImages=12, tSingleImage=None, filePrefix='output', skipIfRun=True, outputFolder='default'):
    # Pops images out of a video.
    # If tSingleImage is defined (time, not frame number), only one image will be popped
    if outputFolder == 'default':
        outputFolder = os.path.dirname(videoPath)
    
    # already written out?
    if not os.path.exists(os.path.join(outputFolder, filePrefix + '_0.jpg')) or not skipIfRun: 
        if tSingleImage is not None: # pop single image at time value
            CMD = ('ffmpeg -loglevel error -skip_frame nokey -y -ss ' + str(tSingleImage) + ' -i ' + videoPath + 
                   " -qmin 1 -q:v 1 -frames:v 1 -vf select='-eq(pict_type\,I)' " + 
                   os.path.join(outputFolder,filePrefix + '0.png'))
            os.system(CMD)
            outImagePath = os.path.join(outputFolder,filePrefix + '0.png')
           
        else: # pop multiple images from video
            lengthVideo = getVideoLength(videoPath)
            timeImageSamples = np.linspace(1,lengthVideo-1,nImages) # disregard first and last second
            for iFrame,t_image in enumerate(timeImageSamples):
                CMD = ('ffmpeg -loglevel error -skip_frame nokey -ss ' + str(t_image) + ' -i ' + videoPath + 
                       " -qmin 1 -q:v 1 -frames:v 1 -vf select='-eq(pict_type\,I)' " + 
                       os.path.join(outputFolder,filePrefix) + '_' + str(iFrame) + '.jpg')
                os.system(CMD)
                outImagePath = os.path.join(outputFolder,filePrefix) + '0.jpg'
                
    return outImagePath


def getVideoLength(filename):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filename],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return float(result.stdout)


def getVideoRotation(videoPath):
    meta = ffmpeg.probe(videoPath)
    try:
        rotation = meta['format']['tags']['com.apple.quicktime.video-orientation']
    except:
        # For AVI (after we rewrite video), no rotation paramter, so just using h and w. 
        # For now this is ok, we don't need leaning right/left for this, just need to know
        # how to orient the pose estimation resolution parameters.
        try: 
            if meta['format']['format_name'] == 'avi':
                if meta['streams'][0]['height']>meta['streams'][0]['width']:
                    rotation = 90
                else:
                    rotation = 0
            else:
                raise Exception('no rotation info')
        except:
            rotation = 90 # upright is 90, and intrinsics were captured in that orientation
        
    return int(rotation)


def saveCameraParameters(filename,CameraParams):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        # os.makedirs(filename,exist_ok=True)

    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()
    
    return True


# %% Triangulation
# If you set ignoreMissingMarkers to True, and pass the DISTORTED keypoints
# as keypoints2D, the triangulation will ignore data from cameras that
# returned (0,0) as marker coordinates.
def triangulateMultiview(CameraParamList, points2dUndistorted, 
                          imageScaleFactor=1, useRotationEuler=False,
                          ignoreMissingMarkers=False,selectCamerasMinReprojError = False,
                          ransac = False, keypoints2D=[],confidence=None):
    # create a list of cameras (says sequence in documentation) from CameraParamList
    cameraList = []    
    nCams = len(CameraParamList) 
    nMkrs = np.shape(points2dUndistorted[0])[0]
    
    for camParams in CameraParamList:
        # get rotation matrix
        if useRotationEuler:
            rotMat = cv2.Rodrigues(camParams['rotation_EulerAngles'])[0]
        else:
            rotMat = camParams['rotation']
        
        c = Camera()
        c.set_K(camParams['intrinsicMat'])
        c.set_R(rotMat)
        c.set_t(np.reshape(camParams['translation'],(3,1)))
        cameraList.append(c)
           
   
    # triangulate
    stackedPoints = np.stack(points2dUndistorted)
    pointsInput = []
    for i in range(stackedPoints.shape[1]):
        pointsInput.append(stackedPoints[:,i,0,:].T)
    
    points3d,confidence3d = nview_linear_triangulations(cameraList,pointsInput,weights=confidence)

    
    # Below are some outlier rejection methods
    
    # A slow, hacky way of rejecting outliers, like RANSAC. 
    # Select the combination of cameras that minimize mean reprojection error for all cameras
    if selectCamerasMinReprojError and nCams>2:
        
        # Function definitions
        def generateCameraCombos(nCams):
            comb = [] ;
            k=nCams-1 ;           
            comb.append(tuple(range(nCams)))
            while k>=2:
                comb = comb + list(combinations(np.arange(nCams),k))
                k=k-1
            return comb
        

        # generate a list of all possible camera combinations down to 2
        camCombos = generateCameraCombos(len(cameraList))
        
        # triangulate for all camera combiations
        points3DList = []
        nMkrs = confidence[0].shape[0]
        reprojError = np.empty((nMkrs,len(camCombos)))
                
        for iCombo,camCombo in enumerate(camCombos):
            camList = [cameraList[i] for i in camCombo]
            points2DList = [pts[:,camCombo] for pts in pointsInput]
            conf = [confidence[i] for i in camCombo]
            
            points3DList.append(nview_linear_triangulations(camList,points2DList,weights=conf))
        
            # compute per-marker, confidence-weighted reprojection errors for all camera combinations
            # reprojError[:,iCombo] = calcReprojectionError(camList,points2DList,points3DList[-1],weights=conf)
            reprojError[:,iCombo] = calcReprojectionError(cameraList,pointsInput,points3DList[-1],weights=confidence)

        # select the triangulated point from camera set that minimized reprojection error
        new3Dpoints = np.empty((3,nMkrs))
        for iMkr in range(nMkrs):
            new3Dpoints[:,iMkr] = points3DList[np.argmin(reprojError[iMkr,:])][:,iMkr]
        
        points3d = new3Dpoints
        
    #RANSAC for outlier rejection - on a per-marker basis, not a per-camera basis. Could be part of the problem
    #Not clear that this is helpful 4/23/21
    if ransac and nCams>2:
        nIter = np.round(np.log(.01)/np.log(1-np.power(.75,2))) # log(1 - prob of getting optimal set) / log(1-(n_inliers/n_points)^minPtsForModel)
        #TODO make this a function of image resolution
        errorUB = 20 # pixels, below this reprojection error, another camera gets added. 
        nGoodModel =3  # model must have this many cameras to be considered good
        
        #functions
        def triangulateLimitedCameras(cameraList,pointsInput,confidence,cameraNumList):
            camList = [cameraList[i] for i in cameraNumList]
            points2DList = [pts[:,cameraNumList] for pts in pointsInput]
            conf = [confidence[i] for i in cameraNumList]
            points3D = nview_linear_triangulations(camList,points2DList,weights=conf)
            return points3D
        
        def reprojErrorLimitedCameras(cameraList,pointsInput,points3D,confidence,cameraNumList):
            if type(cameraNumList) is not list: cameraNumList = [cameraNumList]
            camList = [cameraList[i] for i in cameraNumList]
            points2DList = [pts[:,cameraNumList] for pts in pointsInput]
            conf = [confidence[i] for i in cameraNumList]
            reprojError = calcReprojectionError(camList,points2DList,points3D,weights=conf)
            return reprojError

        
        #initialize
        bestReprojError = 1000 * np.ones(nMkrs) # initial, large value
        best3Dpoint = np.empty((points3d.shape))
        camComboList = [[] for _ in range(nMkrs)]
        
        for iIter in range(int(nIter)):
            np.random.seed(iIter)
            camCombo = np.arange(nCams)
            np.random.shuffle(camCombo) # Seed setting should give same combos every run
            maybeInliers = list(camCombo[:2])
            alsoInliers = [[] for _ in range(nMkrs)]
            
            
            # triangulate maybe inliers
            points3D = triangulateLimitedCameras(cameraList,pointsInput,confidence,maybeInliers)
            
            #error on next camera
            for iMkr in range(nMkrs):
                for j in range(nCams-2):
                    er = reprojErrorLimitedCameras(cameraList,pointsInput,points3D,confidence,camCombo[2+j])
                    # print(er[iMkr])
                    if er[iMkr] < errorUB:
                        alsoInliers[iMkr].append(camCombo[2+j])
                # see if error is bigger than previous, if not, use this combo to triangulate
                # just 1 marker
                if (len(maybeInliers) + len(alsoInliers[iMkr])) >= nGoodModel:
                    thisConf = [np.atleast_1d(c[iMkr]) for c in confidence]
                    point3D = triangulateLimitedCameras(cameraList,[pointsInput[iMkr]],thisConf,maybeInliers + alsoInliers[iMkr])
                    er3D = reprojErrorLimitedCameras(cameraList,[pointsInput[iMkr]],point3D,thisConf,maybeInliers + alsoInliers[iMkr])    
                    # if er3D<bestReprojError[iMkr]:
                    if (len(maybeInliers) + len(alsoInliers[iMkr])) > len(camComboList[iMkr]):
                        best3Dpoint[:,iMkr] = point3D.T
                        bestReprojError[iMkr] = er3D
                        camComboList[iMkr] = maybeInliers.copy() + alsoInliers[iMkr].copy()

        points3d = best3Dpoint
                                    
    
    if ignoreMissingMarkers and nCams>2:        
        # For markers that were not identified by certain cameras,
        # we re-compute their 3D positions but only using cameras that could
        # identify them (ie cameras that did not return (0,0) as coordinates).
        missingCams, missingMarkers = getMissingMarkersCameras(keypoints2D)     
    
        for missingMarker in np.unique(missingMarkers):
            idx_missingMarker = np.where(missingMarkers == missingMarker)[0]
            idx_missingCam = missingCams[idx_missingMarker]
            
            idx_viewedCams = list(range(0, len(cameraList)))
            for i in idx_missingCam:
                idx_viewedCams.remove(i)
                
            CamParamList_viewed = [cameraList[i] for i in idx_viewedCams]
            c_pointsInput = copy.deepcopy(pointsInput)
            for count, pointInput in enumerate(c_pointsInput):
                c_pointsInput[count] = pointInput[:,idx_viewedCams]
            
            points3d_missingMarker = nview_linear_triangulations(
                CamParamList_viewed, c_pointsInput,weights=confidence)
            
            # overwritte marker
            points3d[:, missingMarker] = points3d_missingMarker[:, missingMarker]
    
    return points3d, confidence3d


def nview_linear_triangulations(cameras, image_points,weights=None):
    """
    Computes world coordinates from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param image_points: image coordinates of m correspondences in n views
    :type image_points: sequence of m numpy.ndarray, shape=(2, n)
    :return: m world coordinates
    :rtype: numpy.ndarray, shape=(3, m)
    :weights: numpy.ndarray, shape(nMkrs,nCams)
    """
    assert(type(cameras) == list)
    assert(type(image_points) == list)
    # print(len(cameras), image_points[0].shape[1])
    assert(len(cameras) == image_points[0].shape[1])
    assert(image_points[0].shape[0] == 2)

    world = np.zeros((3, len(image_points)))
    confidence = np.zeros((1,len(image_points)))
    for i, correspondence in enumerate(image_points):
        if weights is not None:
            w = [w[i] for w in weights]
        else:
            w = None
        pt3d, conf = nview_linear_triangulation(cameras, correspondence,weights=w)
        world[:, i] = np.ndarray.flatten(pt3d)
        confidence[0,i] = conf
    return world,confidence


def nview_linear_triangulation(cameras, correspondences,weights = None):
    # print("************")
    # print(cameras[0].P)
    # print(correspondences)
    # print("************")
    """
    Computes ONE world coordinate from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param correspondences: image coordinates correspondences in n views
    :type correspondences: numpy.ndarray, shape=(2, n)
    :return: world coordinate
    :rtype: numpy.ndarray, shape=(3, 1)
    """
    assert(len(cameras) >= 2)
    assert(type(cameras) == list)
    assert(correspondences.shape == (2, len(cameras)))

    def _construct_D_block(P, uv,w=1):
        """
        Constructs 2 rows block of matrix D.
        See [1, p. 88, The Triangulation Problem]
        :param P: camera matrix
        :type P: numpy.ndarray, shape=(3, 4)
        :param uv: image point coordinates (xy)
        :type uv: numpy.ndarray, shape=(2,)
        :return: block of matrix D
        :rtype: numpy.ndarray, shape=(2, 4)
        """

        return w*np.vstack((uv[0] * P[2, :] - P[0, :],
                          uv[1] * P[2, :] - P[1, :]))
    
    # testing weighted least squares
    if weights is None:
        w = np.ones(len(cameras))
        weights = [1 for i in range(len(cameras))]
    else:
        w = [np.nan_to_num(wi,nan=0.5) for wi in weights] # turns nan confidences into 0.5
    
    
    D = np.zeros((len(cameras) * 2, 4))
    for cam_idx, cam, uv in zip(range(len(cameras)), cameras, correspondences.T):
        D[cam_idx * 2:cam_idx * 2 + 2, :] = _construct_D_block(cam.P, uv,w=w[cam_idx])
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    pt3d = p2e(u[:, -1, np.newaxis])
    weightArray = np.asarray(weights)
    if np.count_nonzero(weights)<2:
        # return 0s if there aren't at least 2 cameras with confidence
        pt3d = np.zeros_like(pt3d)
        conf = 0 
    else:
        # if all nan slice (all cameras were splined)
        if all(np.isnan(weightArray[weightArray!=0])):
            conf=.5 # nans get 0.5 confidence
        else:
            conf = np.nanmean(weightArray[weightArray!=0])

    return pt3d,conf


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.
    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)
    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert(type(projective) == np.ndarray)
    assert((projective.shape[0] == 4) | (projective.shape[0] == 3))
    return (projective / projective[-1, :])[0:-1, :]

def getMissingMarkersCameras(keypoints2D):
    # Identify cameras that returned (0,0) as marker coordinates, ie that could
    # not identify the keypoints.
    # missingCams contains the indices of the cameras
    # missingMarkers contains the indices of the markers
    # Eg, missingCams[0] = 5 and missingMarkers[0] = 17 means that the camera
    # with index 5 returned (0,0) as coordinates of the marker with index 17.
    keypoints2D_res = np.reshape(np.stack(keypoints2D), 
                                  (np.stack(keypoints2D).shape[0], 
                                  np.stack(keypoints2D).shape[1], 
                                  np.stack(keypoints2D).shape[3]))
    missingCams, missingMarkers = np.where(np.sum(keypoints2D_res, axis=2) == 0)
    
    return missingCams, missingMarkers


def calcReprojectionError(cameraList,points2D,points3D,weights=None,normalizeError=False):
    reprojError = np.empty((points3D.shape[1],len(cameraList)))
    
    if weights==None:
        weights = [1 for i in range(len(cameraList))]
    for iCam,cam in enumerate(cameraList):
        reproj = cam.world_to_image(points3D)[:2,:]
        this2D = np.array([pt2D[:,iCam] for pt2D in points2D]).T
        reprojError[:,iCam] = np.linalg.norm(np.multiply((reproj-this2D),weights[iCam]),axis=0)
        
        if normalizeError: # Normalize by height of bounding box 
            nonZeroYVals = this2D[1,:][this2D[1,:]>0]
            boxHeight = np.nanmax(nonZeroYVals) - np.nanmin(nonZeroYVals)
            reprojError[:,iCam] /= boxHeight
    weightedReprojError_u = np.mean(reprojError,axis=1)
    return weightedReprojError_u


class Camera:
    """
    Projective camera model
        - camera intrinsic and extrinsic parameters handling
        - various lens distortion models
        - model persistence
        - projection of camera coordinates to an image
        - conversion of image coordinates on a plane to camera coordinates
        - visibility handling
    """

    def __init__(self, id=None):
        """
        :param id: camera identification number
        :type id: unknown or int
        """
        self.K = np.eye(3)  # camera intrinsic parameters
        self.Kundistortion = np.array([])  # could be altered based on K using set_undistorted_view(alpha)
        #  to get undistorted image with all / corner pixels visible
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.kappa = np.zeros((2,))
        self.id = id
        self.size_px = np.zeros((2,))
        # self.size_px_view = np.zeros((2,))  #

        self.bouguet_kc = np.zeros((5,))
        self.kannala_p = np.zeros((6,))
        self.kannala_thetamax = None
        self.division_lambda = 0.
        self.division_z_n = -1
        self.tsai_f = -1
        self.tsai_kappa = -1
        self.tsai_ncx = -1
        self.tsai_nfx = -1
        self.tsai_dx = -1
        self.tsai_dy = -1
        self.opencv_dist_coeff = None
        self.calibration_type = 'standard'  # other possible values: bouguet, kannala, division, opencv
        self.update_P()

    def save(self, filename):
        """
        Save camera model to a YAML file.
        """
        data = {'id': self.id,
                'K': self.K.tolist(),
                'R': self.R.tolist(),
                't': self.t.tolist(),
                'size_px': self.size_px.tolist(),
                'calibration_type': self.calibration_type
                }
        if self.Kundistortion.size != 0:
            data['Kundistortion'] = self.Kundistortion.tolist()
        if self.calibration_type == 'bouguet':
            data['bouguet_kc'] = self.bouguet_kc.tolist()
        elif self.calibration_type == 'kannala':
            data['kannala_p'] = self.kannala_p.tolist()
            data['kannala_thetamax'] = self.kannala_thetamax
        elif self.calibration_type == 'tsai':
            data_tsai = {'tsai_f': self.tsai_f,
                         'tsai_kappa': self.tsai_kappa,
                         'tsai_nfx': self.tsai_nfx,
                         'tsai_dx': self.tsai_dx,
                         'tsai_dy': self.tsai_dy,
                         'tsai_ncx': self.tsai_ncx,
                         }
            data.update(data_tsai)
        elif self.calibration_type == 'division':
            data['division_lambda'] = self.division_lambda
            data['division_z_n'] = self.division_z_n
        elif self.calibration_type == 'opencv' or self.calibration_type == 'opencv_fisheye':
            data['opencv_dist_coeff'] = self.opencv_dist_coeff.tolist()
        else:
            data['kappa'] = self.kappa.tolist()
        yaml.dump(data, open(filename, 'w'))

    def load(self, filename):
        """
        Load camera model from a YAML file.
        Example::
            calibration_type: standard
            K:
            - [1225.2, -7.502186291576686e-14, 480.0]
            - [0.0, 1225.2, 384.0]
            - [0.0, 0.0, 1.0]
            R:
            - [-0.9316877145365, -0.3608289515885, 0.002545329627547]
            - [-0.1725273110187, 0.4247524018287, -0.8888909933995]
            - [0.3296724908378, -0.8263880720441, -0.4579894432589]
            id: 0
            kappa: [0.0, 0.0]
            size_px: [960, 768]
            t:
            - [-1.365061486465]
            - [3.431608806127]
            - [17.74182159488]
        """
        data = yaml.load(open(filename))
        if 'id' in data:
            self.id = data['id']
        if 'K' in data:
            self.K = np.array(data['K']).reshape((3, 3))
        if 'R' in data:
            self.R = np.array(data['R']).reshape((3, 3))
        if 't' in data:
            self.t = np.array(data['t']).reshape((3, 1))
        if 'size_px' in data:
            self.size_px = np.array(data['size_px']).reshape((2,))
        if 'calibration_type' in data:
            self.calibration_type = data['calibration_type']
        if 'Kundistortion' in data:
            self.Kundistortion = np.array(data['Kundistortion'])
        else:
            self.Kundistortion = self.K
        if self.calibration_type == 'bouguet':
            self.bouguet_kc = np.array(data['bouguet_kc']).reshape((5,))
        elif self.calibration_type == 'kannala':
            self.kannala_p = np.array(data['kannala_p']).reshape((6,))
            self.kannala_thetamax = data['kannala_thetamax']  # not used now
            # Focal length actually used is from kannala_p. Why then K is stored? Works for me like this.
            self.K[0, 0] = self.kannala_p[2]
            self.K[1, 1] = self.kannala_p[3]
            # principal point in K and kannala_p[4:] should be consistent
            assert self.K[0, 2] == self.kannala_p[4]
            assert self.K[1, 2] == self.kannala_p[5]
        elif self.calibration_type == 'tsai':
            self.tsai_f = data['tsai_f']
            self.tsai_kappa = data['tsai_kappa']
            self.tsai_ncx = data['tsai_ncx']
            self.tsai_nfx = data['tsai_nfx']
            self.tsai_dx = data['tsai_dx']
            self.tsai_dy = data['tsai_dy']
        elif self.calibration_type == 'division':
            self.division_lambda = data['division_lambda']
            self.division_z_n = data['division_z_n']
        elif self.calibration_type == 'opencv' or self.calibration_type == 'opencv_fisheye':
            self.opencv_dist_coeff = np.array(data['opencv_dist_coeff'])
        elif self.calibration_type == 'standard':
            self.kappa = np.array(data['kappa']).reshape((2,))
        if 'id' not in data and \
                        'K' not in data and \
                        'R' not in data and \
                        't' not in data and \
                        'size_px' not in data and \
                        'calibration_type' not in data and \
                        'Kundistortion' not in data:
            warn('Nothing loaded from %s, check the contents.' % filename)
        self.update_P()

    def update_P(self):
        """
        Update camera P matrix from K, R and t.
        """
        self.P = self.K.dot(np.hstack((self.R, self.t)))

    def set_K(self, K):
        """
        Set K and update P.
        :param K: intrinsic camera parameters
        :type K: numpy.ndarray, shape=(3, 3)
        """
        self.K = K
        self.update_P()

    def set_K_elements(self, u0_px, v0_px, f=1, theta_rad=math.pi/2, a=1):
        """
        Update pinhole camera intrinsic parameters and updates P matrix.
        :param u0_px: principal point x position (pixels)
        :type u0_px: double
        :param v0_px: principal point y position (pixels)
        :type v0_px: double
        :param f: focal length
        :type f: double
        :param theta_rad: digitization raster skew (radians)
        :type theta_rad: double
        :param a: pixel aspect ratio
        :type a: double
        """
        self.K = np.array([[f, -f * 1 / math.tan(theta_rad), u0_px],
                      [0, f / (a * math.sin(theta_rad)), v0_px],
                      [0, 0, 1]])
        self.update_P()

    def set_R(self, R):
        """
        Set camera extrinsic parameters and updates P.
        :param R: camera extrinsic parameters matrix
        :type R: numpy.ndarray, shape=(3, 3)
        """
        self.R = R
        self.update_P()

    def set_R_euler_angles(self, angles):
        """
        Set rotation matrix according to euler angles and updates P.
        :param angles: 3 euler angles in radians,
        :type angles: double sequence, len=3
        """
        rx = angles[0]
        ry = angles[1]
        rz = angles[2]
        from numpy import sin
        from numpy import cos
        self.R = np.array([[cos(ry) * cos(rz),
                            cos(rz) * sin(rx) * sin(ry) - cos(rx) * sin(rz),
                            sin(rx) * sin(rz) + cos(rx) * cos(rz) * sin(ry)],
                           [cos(ry) * sin(rz),
                            sin(rx) * sin(ry) * sin(rz) + cos(rx) * cos(rz),
                            cos(rx) * sin(ry) * sin(rz) - cos(rz) * sin(rx)],
                           [-sin(ry),
                            cos(ry) * sin(rx),
                            cos(rx) * cos(ry)]
                           ])
        self.update_P()

    def set_t(self, t):
        """
        Set camera translation and updates P.
        :param t: camera translation vector
        :type t: numpy.ndarray, shape=(3, 1)
        """
        self.t = t
        self.update_P()

    def get_K_0(self):
        """
        Return ideal calibration matrix (only focal length present).
        :return: ideal calibration matrix
        :rtype: np.ndarray, shape=(3, 3)
        """
        K_0 = np.eye(3)
        K_0[0, 0] = self.get_focal_length()
        K_0[1, 1] = self.get_focal_length()
        return K_0

    def get_A(self, K=None):
        """
        Return part of K matrix that applies center, skew and aspect ratio to ideal image coordinates.
        :rtype: np.ndarray, shape=(3, 3)
        """
        if K is None:
            K = self.K
        A = K.copy()
        A[0, 0] /= self.get_focal_length()
        A[0, 1] /= self.get_focal_length()
        A[1, 1] /= self.get_focal_length()
        return A

    def get_z0_homography(self, K=None):
        """
        Return homography from world plane at z = 0 to image plane.
        :return: 2d plane homography
        :rtype: np.ndarray, shape=(3, 3)
        """
        if K is None:
            K = self.K
        return K.dot(np.hstack((self.R, self.t)))[:, [0, 1, 3]]

    def undistort_image(self, img, Kundistortion=None):
        """
        Transform grayscale image such that radial distortion is removed.
        :param img: input image
        :type img: np.ndarray, shape=(n, m) or (n, m, 3)
        :param Kundistortion: camera matrix for undistorted view, None for self.K
        :type Kundistortion: array-like, shape=(3, 3)
        :return: transformed image
        :rtype: np.ndarray, shape=(n, m) or (n, m, 3)
        """
        if Kundistortion is None:
            Kundistortion = self.K
        if self.calibration_type == 'opencv':
            return cv2.undistort(img, self.K, self.opencv_dist_coeff, newCameraMatrix=Kundistortion)
        elif self.calibration_type == 'opencv_fisheye':
                return cv2.fisheye.undistortImage(img, self.K, self.opencv_dist_coeff, Knew=Kundistortion)
        else:
            xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            img_coords = np.array([xx.ravel(), yy.ravel()])
            y_l = self.undistort(img_coords, Kundistortion)
            if img.ndim == 2:
                return griddata(y_l.T, img.ravel(), (xx, yy), fill_value=0, method='linear')
            else:
                channels = [griddata(y_l.T, img[:, :, i].ravel(), (xx, yy), fill_value=0, method='linear')
                            for i in range(img.shape[2])]
                return np.dstack(channels)

    def undistort(self, distorted_image_coords, Kundistortion=None):
        """
        Remove distortion from image coordinates.
        :param distorted_image_coords: real image coordinates
        :type distorted_image_coords: numpy.ndarray, shape=(2, n)
        :param Kundistortion: camera matrix for undistorted view, None for self.K
        :type Kundistortion: array-like, shape=(3, 3)
        :return: linear image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert distorted_image_coords.shape[0] == 2
        assert distorted_image_coords.ndim == 2
        if Kundistortion is None:
            Kundistortion = self.K
        if self.calibration_type == 'division':
            A = self.get_A(Kundistortion)
            Ainv = np.linalg.inv(A)
            undistorted_image_coords = p2e(A.dot(e2p(self._undistort_division(p2e(Ainv.dot(e2p(distorted_image_coords)))))))
        elif self.calibration_type == 'opencv':
            undistorted_image_coords = cv2.undistortPoints(distorted_image_coords.T.reshape((1, -1, 2)),
                                                           self.K, self.opencv_dist_coeff,
                                                           P=Kundistortion).reshape(-1, 2).T
        elif self.calibration_type == 'opencv_fisheye':
            undistorted_image_coords = cv2.fisheye.undistortPoints(distorted_image_coords.T.reshape((1, -1, 2)),
                                                                   self.K, self.opencv_dist_coeff,
                                                                   P=Kundistortion).reshape(-1, 2).T
        else:
            warn('undistortion not implemented')
            undistorted_image_coords = distorted_image_coords
        assert undistorted_image_coords.shape[0] == 2
        assert undistorted_image_coords.ndim == 2
        return undistorted_image_coords

    def distort(self, undistorted_image_coords, Kundistortion=None):
        """
        Apply distortion to ideal image coordinates.
        :param undistorted_image_coords: ideal image coordinates
        :type undistorted_image_coords: numpy.ndarray, shape=(2, n)
        :param Kundistortion: camera matrix for undistorted coordinates, None for self.K
        :type Kundistortion: array-like, shape=(3, 3)
        :return: distorted image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert undistorted_image_coords.shape[0] == 2
        assert undistorted_image_coords.ndim == 2
        if Kundistortion is None:
            Kundistortion = self.K
        if self.calibration_type == 'division':
            A = self.get_A(Kundistortion)
            Ainv = np.linalg.inv(A)
            distorted_image_coords = p2e(A.dot(e2p(self._distort_division(p2e(Ainv.dot(e2p(undistorted_image_coords)))))))
        elif self.calibration_type == 'opencv':
            undistorted_image_coords_norm = (undistorted_image_coords - column(Kundistortion[0:2, 2])) / \
                                            column(Kundistortion.diagonal()[0:2])
            undistorted_image_coords_3d = np.vstack((undistorted_image_coords_norm,
                                                     np.zeros((1, undistorted_image_coords.shape[1]))))
            distorted_image_coords, _ = cv2.projectPoints(undistorted_image_coords_3d.T, (0, 0, 0), (0, 0, 0),
                                                          self.K, self.opencv_dist_coeff)
            distorted_image_coords = distorted_image_coords.reshape(-1, 2).T
        elif self.calibration_type == 'opencv_fisheye':
            # if self.Kundistortion is not np.array([]):
            #     # remove Kview transformation
            #     undistorted_image_coords = p2e(np.matmul(np.linalg.inv(self.Kundistortion),
            #                                              e2p(undistorted_image_coords)))
            # TODO check correctness
            undistorted_image_coords = p2e(np.matmul(np.linalg.inv(Kundistortion),
                                                     e2p(undistorted_image_coords)))
            distorted_image_coords = cv2.fisheye.distortPoints(undistorted_image_coords.T.reshape((1, -1, 2)),
                                                               self.K, self.opencv_dist_coeff).reshape(-1, 2).T
        else:
            assert False  # not implemented
        assert distorted_image_coords.shape[0] == 2
        assert distorted_image_coords.ndim == 2
        return distorted_image_coords

    def _distort_bouguet(self, undistorted_centered_image_coord):
        """
        Distort centered image coordinate following Bouquet model.
        see http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        :param undistorted_centered_image_coord: linear centered image coordinate(s)
        :type undistorted_centered_image_coord: numpy.ndarray, shape=(2, n)
        :return: distorted coordinate(s)
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert undistorted_centered_image_coord.shape[0] == 2
        kc = self.bouguet_kc
        x = undistorted_centered_image_coord[0, :]
        y = undistorted_centered_image_coord[1, :]
        r_squared = x ** 2 + y ** 2

        # tangential distortion vector
        dx = np.array([2 * kc[2] * x * y + kc[3] * (r_squared + 2 * x ** 2),
                       kc[2] * (r_squared + 2 * y ** 2) + 2 * kc[3] * x * y])
        distorted = (1 + kc[0] * r_squared + kc[1] * r_squared ** 2 + kc[4] * r_squared ** 3) * \
            undistorted_centered_image_coord + dx
        return distorted

    def _distort_kannala(self, camera_coords):
        """
        Distort image coordinate following Kannala model (M6 version only)
        See http://www.ee.oulu.fi/~jkannala/calibration/calibration_v23.tar.gz :genericproj.m
        Juho Kannala, Janne Heikkila and Sami S. Brandt. Geometric camera calibration. Wiley Encyclopedia of Computer Science and Engineering, 2008, page 9.
        :param camera_coords: 3d points in camera coordinates
        :type camera_coords: numpy.ndarray, shape=(3, n)
        :return: distorted metric image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert camera_coords.shape[0] == 3
        x = camera_coords[0, :]
        y = camera_coords[1, :]
        z = camera_coords[2, :]
        k1 = self.kannala_p[0]
        k2 = self.kannala_p[1]

        # angle between ray and optical axis
        theta = np.arccos(z / np.linalg.norm(camera_coords, axis=0))

        # radial projection (Kannala 2008, eq. 17)
        r = k1 * theta + k2 * theta ** 3

        hypotenuse = np.linalg.norm(camera_coords[0:2, :], axis=0)
        hypotenuse[hypotenuse == 0] = 1  # avoid dividing by zero
        image_x = r * x / hypotenuse
        image_y = r * y / hypotenuse
        return np.vstack((image_x, image_y))

    def _undistort_tsai(self, distorted_metric_image_coord):
        """
        Undistort centered image coordinate following Tsai model.
        :param distorted_metric_image_coord: distorted METRIC image coordinates
            (metric image coordiante = image_xy * f / z)
        :type distorted_metric_image_coord: numpy.ndarray, shape=(2, n)
        :return: linear image coordinate(s)
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert distorted_metric_image_coord.shape[0] == 2
        # see http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
        x = distorted_metric_image_coord[0, :]
        y = distorted_metric_image_coord[1, :]
        r_squared = x ** 2 + y ** 2

        undistorted = (1 + self.tsai_kappa * r_squared) * distorted_metric_image_coord
        return undistorted

    def _distort_tsai(self, metric_image_coord):
        """
        Distort centered metric image coordinates following Tsai model.
        See: Devernay, Frederic, and Olivier Faugeras. "Straight lines have to be straight."
        Machine vision and applications 13.1 (2001): 14-24. Section 2.1.
        (only for illustration, the formulas didn't work for me)
        http://www.cvg.rdg.ac.uk/PETS2009/sample.zip :CameraModel.cpp:CameraModel::undistortedToDistortedSensorCoord
        Analytical inverse of the undistort_tsai() function.
        :param metric_image_coord: centered metric image coordinates
            (metric image coordinate = image_xy * f / z)
        :type metric_image_coord: numpy.ndarray, shape=(2, n)
        :return: distorted centered metric image coordinates
        :rtype: numpy.ndarray, shape=(2, n)
        """
        assert metric_image_coord.shape[0] == 2
        x = metric_image_coord[0, :]  # vector
        y = metric_image_coord[1, :]  # vector
        r_u = np.sqrt(x ** 2 + y ** 2)  # vector
        c = 1.0 / self.tsai_kappa  # scalar
        d = -c * r_u  # vector

        # solve polynomial of 3rd degree for r_distorted using Cardan method:
        # https://proofwiki.org/wiki/Cardano%27s_Formula
        # r_distorted ** 3 + c * r_distorted + d = 0
        q = c / 3.  # scalar
        r = -d / 2.  # vector
        delta = q ** 3 + r ** 2  # polynomial discriminant, vector

        positive_mask = delta >= 0
        r_distorted = np.zeros((metric_image_coord.shape[1]))

        # discriminant > 0
        s = cbrt(r[positive_mask] + np.sqrt(delta[positive_mask]))
        t = cbrt(r[positive_mask] - np.sqrt(delta[positive_mask]))
        r_distorted[positive_mask] = s + t

        # discriminant < 0
        delta_sqrt = np.sqrt(-delta[~positive_mask])
        s = cbrt(np.sqrt(r[~positive_mask] ** 2 + delta_sqrt ** 2))
        # s = cbrt(np.sqrt(r[~positive_mask] ** 2 + (-delta[~positive_mask]) ** 2))
        t = 1. / 3 * np.arctan2(delta_sqrt, r[~positive_mask])
        r_distorted[~positive_mask] = -s * np.cos(t) + s * np.sqrt(3) * np.sin(t)

        return metric_image_coord * r_distorted / r_u
        
    def _undistort_division(self, z_r):
        """
        Undistort centered image coordinate(s) following the division model.
        :param z_r: radially distorted centered image coordinate(s)
        :type z_r: numpy.ndarray, shape(2, n)
        
        :return: linear image coordinate(s)
        :rtype: numpy.ndarray, shape(2, n)        
        """
        assert (-1 < self.division_lambda < 1)
        return (1 - self.division_lambda) / \
               (1 - self.division_lambda * np.sum(z_r ** 2, axis=0) / self.division_z_n ** 2) * z_r

    def _distort_division(self, z_l):
        """
        Distort centered image coordinate(s) following the division model.
        :param z_l: linear centered image coordinate(s)
        :type z_l: numpy.ndarray, shape(2, n)
        :return: radially distorted image coordinate(s)
        :rtype: numpy.ndarray, shape(2, n)
        """
        z_hat = 2 * z_l / (1 - self.division_lambda)
        return z_hat / (1 + np.sqrt(1 + self.division_lambda * np.sum(z_hat ** 2, axis=0) /
                                    np.sum(self.division_z_n ** 2, axis=0)))
    
    def get_focal_length(self):
        """
        Get camera focal length.
        :return: focal length
        :rtype: double
        """
        return self.K[0, 0]

    def get_principal_point_px(self):
        """
        Get camera principal point.
        :return: x and y pixel coordinates
        :rtype: numpy.ndarray, shape=(1, 2)
        """
        return self.K[0:2, 2].reshape((1, 2))

    def is_visible(self, xy_px):
        """
        Check visibility of image points.
        :param xy_px: image point(s)
        :type xy_px: np.ndarray, shape=(2, n)
        :return: visibility of image points
        :rtype: numpy.ndarray, shape=(1, n), dtype=bool
        """
        assert xy_px.shape[0] == 2
        return (xy_px[0, :] >= 0) & (xy_px[1, :] >= 0) & \
               (xy_px[0, :] < self.size_px[0]) & \
               (xy_px[1, :] < self.size_px[1])

    def is_visible_world(self, world):
        """
        Check visibility of world points.
        :param world: world points
        :type world: numpy.ndarray, shape=(3, n)
        :return: visibility of world points
        :rtype: numpy.ndarray, shape=(1, n), dtype=bool
        """
        assert world.shape[0] == 3
        xy_px = p2e(self.world_to_image(world))
        return self.is_visible(xy_px)

    def get_camera_center(self):
        """
        Returns camera center in the world coordinates.
        :return: camera center in projective coordinates
        :rtype: np.ndarray, shape=(4, 1)
        """
        return self._null(self.P)

    def world_to_image(self, world):
        """
        Project world coordinates to image coordinates.
        :param world: world points in 3d projective or euclidean coordinates
        :type world: numpy.ndarray, shape=(3 or 4, n)
        :return: projective image coordinates
        :rtype: numpy.ndarray, shape=(3, n)
        """
        assert(type(world) == np.ndarray)
        if self.calibration_type == 'opencv' or self.calibration_type == 'opencv_fisheye':
            if world.shape[0] == 4:
                world = p2e(world)
            if self.calibration_type == 'opencv':
                distorted_image_coords = cv2.projectPoints(world.T, self.R, self.t,
                                                           self.K, self.opencv_dist_coeff)[0].reshape(-1, 2).T
            else:
                distorted_image_coords = cv2.fisheye.projectPoints(
                        world.T.reshape((1, -1, 3)), cv2.Rodrigues(self.R)[0],
                        self.t, self.K, self.opencv_dist_coeff)[0].reshape(-1, 2).T
            return e2p(distorted_image_coords)
        if world.shape[0] == 3:
            world = e2p(world)
        camera_coords = np.hstack((self.R, self.t)).dot(world)
        if self.calibration_type == 'bouguet':
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_metric = xy / z
            image_coords_distorted_metric = self._distort_bouguet(image_coords_metric)
            return self.K.dot(e2p(image_coords_distorted_metric))
        elif self.calibration_type == 'tsai':
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_metric = xy * self.tsai_f / z
            image_coords_distorted_metric = self._distort_tsai(image_coords_metric)
            return self.K.dot(e2p(image_coords_distorted_metric))
        elif self.calibration_type == 'kannala':
            image_coords_distorted_metric = self._distort_kannala(camera_coords)
            return self.K.dot(e2p(image_coords_distorted_metric))
        elif self.calibration_type == 'division':
            # see [1, page 54]
            return self.get_A().dot(e2p(self._distort_division(p2e(self.get_k0().dot(camera_coords)))))
        else:
            xy = camera_coords[0:2, :]
            z = camera_coords[2, :]
            image_coords_distorted_metric = xy / z
            return self.K.dot(e2p(image_coords_distorted_metric))

    def image_to_world(self, image_px, z):
        """
        Project image points with defined world z to world coordinates.
        :param image_px: image points
        :type image_px: numpy.ndarray, shape=(2 or 3, n)
        :param z: world z coordinate of the projected image points
        :type z: float
        :return: n projective world coordinates
        :rtype: numpy.ndarray, shape=(3, n)
        """
        if image_px.shape[0] == 3:
            image_px = p2e(image_px)
        image_undistorted = self.undistort(image_px)
        tmpP = np.hstack((self.P[:, [0, 1]], self.P[:, 2, np.newaxis] * z + self.P[:, 3, np.newaxis]))
        world_xy = p2e(np.linalg.inv(tmpP).dot(e2p(image_undistorted)))
        return np.vstack((world_xy, z * np.ones(image_px.shape[1])))

    def get_view_matrix(self, alpha):
        """
        Returns camera matrix for handling image and coordinates distortion and undistortion. Based on alpha,
        up to all pixels of the distorted image can be visible in the undistorted image.
        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1
                      (when all the source image pixels are retained in the undistorted image). For convenience for -1
                      returns custom camera matrix self.Kundistortion and None returns self.K.
        :type alpha: float or None
        :return: camera matrix for a view defined by alpha
        :rtype: array, shape=(3, 3)
        """
        if alpha == -1:
            Kundistortion = self.Kundistortion
        elif alpha is None:
            Kundistortion = self.K
        elif self.calibration_type == 'opencv':
            Kundistortion, _ = cv2.getOptimalNewCameraMatrix(self.K, self.opencv_dist_coeff, tuple(self.size_px), alpha)
        elif self.calibration_type == 'opencv_fisheye':
            Kundistortion = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.opencv_dist_coeff,
                                                                                   tuple(self.size_px), self.R,
                                                                                   balance=alpha)
        else:
            # TODO
            assert False, 'not implemented'
        return Kundistortion

    def plot_world_points(self, points, plot_style, label=None,
                          solve_visibility=True):
        """
        Plot world points to a matplotlib figure.
        :param points: world points (projective or euclidean)
        :type points: numpy.ndarray, shape=(3 or 4, n) or list of lists
        :param plot_style: matplotlib point and line style code, e.g. 'ro'
        :type plot_style: str
        :param label: label plotted under points mean
        :type label: str
        :param solve_visibility: if true then plot only if all points are visible
        :type solve_visibility: bool
        """
        object_label_y_shift = +25
        import matplotlib.pyplot as plt

        if type(points) == list:
            points = np.array(points)
        points = np.atleast_2d(points)
        image_points_px = p2e(self.world_to_image(points))
        if not solve_visibility or np.all(self.is_visible(image_points_px)):
            plt.plot(image_points_px[0, :],
                     image_points_px[1, :], plot_style)
            if label:
                    max_y = max(image_points_px[1, :])
                    mean_x = image_points_px[0, :].mean()
                    plt.text(mean_x, max_y + object_label_y_shift, label)

    def _null(self, A, eps=1e-15):
        """
        Matrix null space.
        For matrix null space holds: A * null(A) = zeros
        source: http://mail.scipy.org/pipermail/scipy-user/2005-June/004650.html
        :param A: input matrix
        :type A: numpy.ndarray, shape=(m, n)
        :param eps: values lower than eps are considered zero
        :type eps: double
        :return: null space of the matrix A
        :rtype: numpy.ndarray, shape=(n, 1)
        """
        u, s, vh = np.linalg.svd(A)
        n = A.shape[1]   # the number of columns of A
        if len(s) < n:
            expanded_s = np.zeros(n, dtype=s.dtype)
            expanded_s[:len(s)] = s
            s = expanded_s
        null_mask = (s <= eps)
        null_space = np.compress(null_mask, vh, axis=0)
        return np.transpose(null_space)


def e2p(euclidean):
    """
    Convert 2d or 3d euclidean to projective coordinates.
    :param euclidean: projective coordinate(s)
    :type euclidean: numpy.ndarray, shape=(2 or 3, n)
    :return: projective coordinate(s)
    :rtype: numpy.ndarray, shape=(3 or 4, n)
    """
    assert(type(euclidean) == np.ndarray)
    assert((euclidean.shape[0] == 3) | (euclidean.shape[0] == 2))
    return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


def column(vector):
    """
    Return column vector.
    :param vector: np.ndarray
    :return: column vector
    :rtype: np.ndarray, shape=(n, 1)
    """
    return vector.reshape((-1, 1))


def isCheckerboardUpsideDown(CameraParams):
    # With backwall orientation, R[1,1] will always be positive in correct orientation
    # and negative if upside down
    for cam in list(CameraParams.keys()):
        if CameraParams[cam] is not None:
            upsideDown = CameraParams[cam]['rotation'][1,1] < 0
            break
        #Default if no camera params (which is a garbage case anyway)
        upsideDown = False

    return upsideDown


def writeTRCfrom3DKeypoints(keypoints3D, pathOutputFile, keypointNames, 
                            frameRate=60, rotationAngles={}):
    
    keypoints3D_res = np.empty((keypoints3D.shape[2],
                                keypoints3D.shape[0]*keypoints3D.shape[1]))
    for iFrame in range(keypoints3D.shape[2]):
        keypoints3D_res[iFrame,:] = np.reshape(
            keypoints3D[:,:,iFrame], 
            (1,keypoints3D.shape[0]*keypoints3D.shape[1]),"F")
    
    # Change units to save data in m.
    keypoints3D_res /= 1000
    
    # Do not write face markers, they are unreliable and useless.
    faceMarkers = getOpenPoseFaceMarkers()[0]
    idxFaceMarkers = [keypointNames.index(i) for i in faceMarkers]    
    idxToRemove = np.hstack([np.arange(i*3,i*3+3) for i in idxFaceMarkers])
    keypoints3D_res_sel = np.delete(keypoints3D_res, idxToRemove, axis=1)  
    keypointNames_sel = [i for i in keypointNames if i not in faceMarkers]

    with open(pathOutputFile,"w") as f:
        numpy2TRC(f, keypoints3D_res_sel, keypointNames_sel, fc=frameRate, 
                  units="m")
    
    # Rotate data to match OpenSim conventions; this assumes the chessboard
    # is behind the subject and the chessboard axes are parallel to those of
    # OpenSim.
    trc_file = utilsDataman.TRCFile(pathOutputFile)    
    for axis,angle in rotationAngles.items():
        trc_file.rotate(axis,angle) #NWM czy potrzebne

    trc_file.write(pathOutputFile)   
    
    return None

def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count+1] = header 
        
    # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())
    
    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    
    num_frames=data.shape[0]
    num_markers=len(header_mapping.keys())
    
    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            fc, fc, num_frames,
            num_markers, units, fc,
            1, num_frames))
    
    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')
    
    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame+1,(frame)/fc+t_start)) # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write("{:.5f}\t{:.5f}\t{:.5f}\t".format(data[frame,0+(key-1)*3], data[frame,1+(key-1)*3], data[frame,2+(key-1)*3]))
        f.write("\n")


def getOpenPoseFaceMarkers():
    
    faceMarkerNames = ['Nose', 'REye', 'LEye', 'REar', 'LEar']
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]
    
    return faceMarkerNames, idxFaceMarkers


def getOpenPoseMarkerNames():
    
    markerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                   "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                   "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    
    return markerNames


def plot_3d(data_3D):
    VecStart_x = []
    VecStart_y = []
    VecStart_z = []
    VecEnd_x = []
    VecEnd_y = []
    VecEnd_z = []

    connections=[(32,30),(30,28),(32,28),(26,28),(26,24),(24,23),(23,25),(25,27),(27,31),(31,29),(29,27),(24,12),(12,11),(11,23),(11,13),(13,15),(15,17),(17,19),(19,15),(15,21),(12,14),(14,16),(16,18),(18,20),(20,16),(16,22),(10,9),(8,6),(6,5),(5,4),(4,0),(0,1),(1,2),(2,3),(3,7),(13,15)]
    for start, end in connections:
        VecStart_x.append(data_3D[start*3::99][0])
        VecStart_y.append(data_3D[start*3+1::99][0])
        VecStart_z.append(data_3D[start*3+2::99][0])
        VecEnd_x.append(data_3D[end*3::99][0])
        VecEnd_y.append(data_3D[end*3+1::99][0])
        VecEnd_z.append(data_3D[end*3+2::99][0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(VecEnd_z)):
        ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

    plt.savefig("Data/plot.png")
    plt.close()  


def return_max_val(keypoints3D):
    x_min=min(keypoints3D[0::3])
    y_min=min(keypoints3D[1::3])
    z_min=min(keypoints3D[2::3])
    keypoints3D[0::3]=[elem-x_min for elem in keypoints3D[0::3]]
    keypoints3D[1::3]=[elem-y_min for elem in keypoints3D[1::3]]
    keypoints3D[2::3]=[elem-z_min for elem in keypoints3D[2::3]]
    max_val=max(keypoints3D)
    return max_val, x_min, y_min, z_min

def return_normalized_keypoints(keypoints3D, max_val, x_min, y_min, z_min):
    keypoints3D=sum([[(keypoints3D[0][ind][0]-x_min)/max_val, (keypoints3D[1][ind][0]-y_min)/max_val, (keypoints3D[2][ind][0]-z_min)/max_val] for ind in range(len(keypoints3D[0]))],[])    
    # TODO obracanie x, y, z
    return keypoints3D

def all_transformations(keypoints3D, translation=None):
    if translation:
        perm, flip = translation
        augmented_keypoints3D = [0] * len(keypoints3D)
        for i in range(0, len(keypoints3D), 3):
            augmented_keypoints3D[perm[0] + i] = 1 - keypoints3D[2 + i] if flip[0] else keypoints3D[2 + i]
            augmented_keypoints3D[perm[1] + i] = 1 - keypoints3D[0 + i] if flip[1] else keypoints3D[0 + i]
            augmented_keypoints3D[perm[2] + i] = 1 - keypoints3D[1 + i] if flip[2] else keypoints3D[1 + i]
        return augmented_keypoints3D
    else:
        permutations = list(itertools.permutations([0, 1, 2]))
        flip_operations = [[False, False, False], [True, False, False], [False, True, False],
                        [False, False, True], [True, True, False], [True, False, True],
                        [False, True, True], [True, True, True]]
        
        for perm in permutations:
            for flip in flip_operations:
                a = [0] * len(keypoints3D)
                for i in range(0, len(keypoints3D), 3):
                    a[perm[0] + i] = 1 - keypoints3D[2 + i] if flip[0] else keypoints3D[2 + i]
                    a[perm[1] + i] = 1 - keypoints3D[0 + i] if flip[1] else keypoints3D[0 + i]
                    a[perm[2] + i] = 1 - keypoints3D[1 + i] if flip[2] else keypoints3D[1 + i]
                x_corr = a[0::3][32] - a[0::3][30]>0 and a[0::3][31] - a[0::3][29]>0 and a[0::3][0]-a[0::3][12]>0 and a[0::3][0]-a[0::3][11]>0
                y_corr = a[1::3][11] - a[1::3][12]>0 and a[1::3][23] - a[1::3][24]>0 and a[1::3][29] - a[1::3][30]>0 and a[1::3][31] - a[1::3][32]>0
                z_corr = a[2::3][0] - a[2::3][30]>0 and a[2::3][12] - a[2::3][24]>0 and a[2::3][11] - a[2::3][23]>0 and a[2::3][23] - a[2::3][25]>0
                if x_corr and y_corr and z_corr:
                    return keypoints3D, (perm, flip)
        print("Nie znaleziono przekształcenia!")
        return None, (None, None)

def record_video(caps):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Użyj kodeka MJPEG
    image_list=[]
    frame_width_list=[]
    frame_height_list=[]

    image_list=[cap.read() for ind, cap in enumerate(caps)]
    frame_width_list=[int(cap.get(3)) for ind, cap in enumerate(caps)]
    frame_height_list=[int(cap.get(4)) for ind, cap in enumerate(caps)]
    output_files_list = [cv2.VideoWriter(f'Data/Cam{ind}/calibration_{ind}.mov', fourcc, fps, (frame_width_list[ind], frame_height_list[ind])) for ind, frame in enumerate(image_list)]
    clock_ticks=0
    timer=time.time()
    while True:
        image_list=[cap.read() for ind, cap in enumerate(caps)]
        [output_file.write(image_list[ind][1]) for ind, output_file in enumerate(output_files_list)]
        if time.time()-timer>clock_ticks:
            os.system('cls')
            print(5-clock_ticks)
            clock_ticks=clock_ticks+1
        if time.time()-timer>5:
            [cap.release() for cap in caps]
            [out.release() for out in output_files_list]
            cv2.destroyAllWindows()
            break
        
def create_metadata(cams, square_h, square_w, square_mm):
    data = {
        "augmentermodel": "v0.3",
        "calibrationSettings": {
            "overwriteDeployedIntrinsics": False,
            "saveSessionIntrinsics": False
        },
        "checkerBoard": {
            "black2BlackCornersHeight_n": square_h,
            "black2BlackCornersWidth_n": square_w,
            "placement": "backWall",
            "squareSideLength_mm": square_mm
        },
        "filterfrequency": "default",
        "gender_mf": "m",
        "height_m": 1.85,
        "iphoneModel": cams,
        "markerAugmentationSettings": {
            "markerAugmenterModel": "LSTM"
        },
        "mass_kg": 76.0,
        "openSimModel": "LaiUhlrich2022",
        "posemodel": "hrnet",
        "subjectID": "user"
    }

    with open("utils/sessionMetadata.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

def make_dirs():
    for path in ["Data", "Data/Videos", "Data/Videos/Cam0", "Data/Videos/Cam1", "Data/Videos/Cam2", "Data/Videos/Cam3"]:
        if not os.path.exists(path):
            os.mkdir(path) 

    # for id, camName in enumerate(sessionMetadata['iphoneModel']):
    #     path=f"Data/Videos/{camName}"
    #     if not os.path.exists(path):
    #         os.mkdir(path) 

    #     path=f"Data/Videos/{camName}/calibration"
    #     if not os.path.exists(path):
    #         os.mkdir(path) 

