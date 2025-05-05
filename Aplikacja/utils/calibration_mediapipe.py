# import os
import yaml
# import cv2 as cv
# import pickle
# import time
# from utils.utils import rotateIntrinsics, calcExtrinsicsFromVideo, saveCameraParameters
from utils.utils import record_video, make_dirs
# from IPython.display import clear_output
# import mediapipe as mp
# import numpy as np
# from utils.utils import return_max_val, return_normalized_keypoints, all_transformations, triangulateMultiview


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


def calibrate_camera(caps):
    record_video(caps)

# # wczytywanie pliku sessionMetadata.yaml
# sessionMetadata_path="init/sessionMetadata.yaml"
# myYamlFile = open(sessionMetadata_path)
# sessionMetadata = yaml.load(myYamlFile, Loader=yaml.FullLoader)

# # parametry szachownicy
# CheckerBoardParams = {
#     'dimensions': (
#         sessionMetadata['checkerBoard']['black2BlackCornersWidth_n'],
#         sessionMetadata['checkerBoard']['black2BlackCornersHeight_n']),
#     'squareSize': 
#         sessionMetadata['checkerBoard']['squareSideLength_mm']}   

# # tworzenie potrzebnych folderow
# make_dirs(sessionMetadata)

# # Tymczasowe rozwiazanie
# # devides=["old_videos/calibration_0.mov","old_videos/calibration_1.mov"] #load camera or video
# devides=[0, 1] #load camera or video
# caps =[cv.VideoCapture(file) for file in devides]#index kamery lub sciezka do zdjecia
# record_video(caps, sessionMetadata)

# # Kalibracja kazdej kamery
# CamParamDict = {}
# for id, camName in enumerate(sessionMetadata['iphoneModel']):
#     # Interinsics 
#     permIntrinsicDir = f"CameraIntrinsics/{sessionMetadata['iphoneModel'][camName]}/Deployed/cameraIntrinsics.pickle"
#     open_file = open(permIntrinsicDir, "rb")
#     CamParams = pickle.load(open_file)
#     open_file.close()
#     extrinsicPath=f"Data/Videos/{camName}/calibration/calibration_{id}.mov"
#     CamParams = rotateIntrinsics(CamParams, extrinsicPath)

#     # Extrinsics
#     CamParams = calcExtrinsicsFromVideo(
#         extrinsicPath,CamParams, CheckerBoardParams, 
#         visualize=False, imageUpsampleFactor=4,
#         useSecondExtrinsicsSolution = None)

#     # wyswietl zdjecia do kalibracji
#     img = cv.imread(f"Data/Videos/{camName}/calibration/extrinsicImage0.png", cv.IMREAD_COLOR)
#     cv.imshow("image", img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

#     # zapisz parametry
#     if CamParams is not None:
#         CamParamDict[camName] = CamParams.copy()
#     else:
#         CamParamDict[camName] = None

# print("Creating cameraIntrinsicsExtrinsics.pickle")
# for camName in CamParamDict:
#     saveCameraParameters( f"Data/Videos/{camName}/cameraIntrinsicsExtrinsics.pickle", 
#         CamParamDict[camName])
    
# # usuwanie niepotrzebnych folderow
# import shutil
# shutil.rmtree("Data/CalibrationImages")
# os.system('cls')
# print("Checkboard calibration complited!")
# user_input=input("Zacząć kalibrować neutral pose?\n'y' - yes\n")
# if user_input == "y":
#     os.system('cls')
# else:
#     raise("Zatrzymano")

# ###############################################################################################################################
# ################################################## Kalibracja neutral pose, max_val oraz obrotu osi ###########################
# ###############################################################################################################################    
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# points2d=[0 for id, camName in enumerate(sessionMetadata['iphoneModel'])]
# landmarks_on_screen=[False for landmarks in ["init/calibration_0.mov","init/calibration_1.mov"]]
# CameraParamList_selectedCams=[0 for id, camName in enumerate(sessionMetadata['iphoneModel'])]
# keypoints2D={}
# thisConfidence={}
# calibrate_env=True 
# reset_timer=True
# max_val, x_min, y_min, z_min=1, 0, 0, 0
# caps =[cv.VideoCapture(file) for file in devides]#index kamery lub sciezka do zdjecia

# with mp_pose.Pose(
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7,
#     static_image_mode=True,
#     model_complexity=2) as pose:
#     cap=caps[0]
#     while cap.isOpened():
#         for id, camName in enumerate(sessionMetadata['iphoneModel']):
#             cap=caps[id]
#             success, image = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame.")
#                 break
#             image.flags.writeable = False
#             image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#             results = pose.process(image)
#             image.flags.writeable = True
#             image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#             cv.imwrite(f"Data/Videos/{camName}/calibration/frame.jpg", image)
#             landmarks_on_screen[id] = results.pose_landmarks!=None
#             if results.pose_landmarks!=None:
#                 image_height, image_width, _ = image.shape
#                 #2D keypoints
#                 keypoints=sum([[results.pose_landmarks.landmark[ind].x*image_width, results.pose_landmarks.landmark[ind].y*image_width, results.pose_landmarks.landmark[ind].visibility] for ind in range(33)],[])
#                 data={"version":1.3,"people":[{"person_id":[-1],"pose_keypoints_2d":keypoints,"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
#                 keypoints2D[camName]=np.array(data['people'][0]['pose_keypoints_2d']).reshape(33, 3)[:,:2].reshape(33,1, 2)
#                 thisConfidence[camName]=np.array(data['people'][0]['pose_keypoints_2d']).reshape(33, 3)[:,2]

#             path=f"Data/Videos/{camName}/cameraIntrinsicsExtrinsics.pickle"
#             open_file = open(path, "rb")
#             CamParams = pickle.load(open_file)
#             open_file.close()   
#             CameraParamList_selectedCams[id]=CamParams
#             ignoreMissingMarkers=False
#         if 0 not in landmarks_on_screen:
#             if calibrate_env:
#                 if reset_timer==True:
#                     timer=time.time()
#                     reset_timer=False
#                 os.system('cls')
#                 print(f"Make neutral pose in: {5+(timer-time.time())}")
#                 if time.time()-timer>5:
#                     points2d=list(keypoints2D.values())
#                     keypoints3D, confidence3D=np.ones((3,33,1)), np.ones((1,33,1))
#                     keypoints3D[:,:,0], confidence3D[:,:,0] = triangulateMultiview(CameraParamList_selectedCams, points2d, imageScaleFactor=1, useRotationEuler=False,ignoreMissingMarkers=ignoreMissingMarkers, keypoints2D=[],confidence=list(thisConfidence.values()))
#                     keypoints3D = return_normalized_keypoints(keypoints3D.tolist(), max_val, x_min, y_min, z_min)
#                     max_val, x_min, y_min, z_min=return_max_val(keypoints3D)
#                     keypoints3D, translation = all_transformations(keypoints3D)
#                     calibrate_env=False
#                     break
#         else:
#             reset_timer=True #jesli przerwie dzialanie modelu zresetuj calibracje srodowiska

# # Sava calibration file
# sessionMetadata.update({
#     'max_val': max_val, 
#     'x_min, y_min, z_min': (x_min, y_min, z_min),
#     'translation': translation})

# with open('Data/Videos/sessionMetadata.yaml', 'w') as file:
#     yaml.dump(sessionMetadata, file, default_flow_style=False)

# os.system('cls')
# print(f"""max_val: {max_val}
# x_min, y_min, z_min: {x_min},  {y_min},  {z_min}
# translation: {translation}""")
# for cap in caps:
#     cap.release()