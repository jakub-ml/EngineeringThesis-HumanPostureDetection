import gradio as gr
import numpy as np
from PIL import Image
import time
import cv2
from utils.utils import create_metadata, record_video, rotateIntrinsics, calcExtrinsicsFromVideo, saveCameraParameters, triangulateMultiview, return_normalized_keypoints, return_max_val, all_transformations, make_dirs, plot_3d
import yaml
import os
import pickle
import mediapipe as mp
import shutil
import matplotlib.image as mpimg
# import run_mediapipe
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

image_global = np.zeros([10,10,3], dtype=np.uint8)
state = "stop"
model = load_model("GRU_40_64_20_aplikacja_model.h5")
scaler = StandardScaler()
scaler.mean_ = np.array([ 2.49446126,  0.35554637,  0.59303074,  1.0385519 ,  0.28357831,
        0.30178513,  0.90482986,  0.28215475,  0.2779665 ,  1.61682663,
        0.40081268,  0.42749397,  1.21664909,  0.31294344,  0.35804453,
        0.89330182,  0.26190105,  0.28300911,  1.0822468 ,  0.27157724,
        0.32185278, 13.6805443 ,  2.42257751,  3.22244742,  1.07000676,
        0.26271298,  0.30618204,  1.35416284,  0.29591056,  0.35182584,
        3.71081797,  0.53280135,  0.72044774,  7.77245033,  1.22413307,
        1.78434738,  5.04779756,  0.45631245,  0.9763854 ,  5.20252256,
        0.35008625,  1.40058953,  1.98317605,  0.69613097,  0.27613606,
       11.56912296,  0.47235022,  3.10689712,  8.96189377,  1.27617224,
        1.35963383, 27.58869349,  1.42479599,  2.08755953, 18.8344391 ,
        3.51584743,  2.42753988, 19.09473477,  2.98864323,  2.62769185,
        7.68021544,  1.1766933 ,  1.90516793,  4.01621961,  0.42532167,
        1.04799875, 28.54306975,  4.15884143,  2.04241436,  4.12913013,
        0.83960968,  0.38607713,  1.07166485,  0.19499686,  0.17271223,
        1.81377538,  0.38314513,  0.2809106 ,  5.15392713,  0.74558767,
        0.66191817, 13.62636678,  2.1238507 ,  1.1420112 , 10.83189232,
        1.85964334,  0.86858114, 48.07205078,  1.48182308,  3.29145195,
       18.6778941 ,  4.96377387,  2.09331418,  7.10021621,  1.16220148,
        0.6465051 , 39.68050084,  3.78786288,  4.32110681])

scaler.scale_ = np.array([ 1.14395238,  0.2466488 ,  0.80307759,  1.17645265,  0.24936065,
        0.81508349,  1.17021584,  0.25323272,  0.81761499,  1.17745531,
        0.25365913,  0.81739895,  1.16158421,  0.2443256 ,  0.82011149,
        1.16919513,  0.2407582 ,  0.81986432,  1.17137881,  0.23815458,
        0.82130121,  1.26286529,  0.23972148,  0.80182755,  1.1688231 ,
        0.23365641,  0.8268288 ,  1.14368029,  0.24871904,  0.78750589,
        1.12483445,  0.24133512,  0.79112845,  1.0918459 ,  0.28051633,
        0.74262816,  1.07405479,  0.2037395 ,  0.74874443,  0.98151313,
        0.29753138,  0.59985475,  0.92887634,  0.19265528,  0.60332539,
        0.99457562,  0.28395517,  0.44806567,  0.89185705,  0.20128074,
        0.45390455,  1.03033808,  0.30444786,  0.42687283,  1.08168934,
        0.17546043,  0.4026003 ,  0.77272567,  0.30677558,  0.4518919 ,
        0.88879939,  0.19935378,  0.42488769,  0.86610813,  0.27760691,
        0.4551112 ,  0.9580882 ,  0.17752869,  0.43990206,  0.91413083,
        0.23949366,  0.48705419,  0.86456173,  0.19554373,  0.49692476,
        0.8835124 ,  0.24400615,  0.24746503,  0.87229762,  0.17381238,
        0.24607715,  0.79204097,  0.21869983,  0.0547836 ,  0.81360726,
        0.13116657,  0.05323745,  1.22892342,  0.20322272,  0.05809654,
        0.93024975,  0.11595687,  0.01964351,  0.86730053,  0.18685386,
        0.00589425,  0.65895619,  0.14769997, -0.04777756])

sample_size=40
keypoints3D_l=[0 for elem in range(40)]

# def run_model(points_list):

def start_stop():
    global state
    if state=="start":
        state =  "stop"
    elif state=="stop":
        state = "start"

def sentence_builder(slider, camera_1, camera_2, camera_3, camera_4):
    cameras=[camera_1, camera_2, camera_3, camera_4]
    all_cameras=f'Dane o {slider} kamerach:\n'
    for id, camera in enumerate(cameras): 
        if camera!=None:
            all_cameras=all_cameras=all_cameras+f"Kamera {id+1}, Model: {camera}\n"
    return all_cameras[:-2]

def camera_parameters(camera_1, camera_2, camera_3, camera_4, square_h, square_w, square_mm):
    cameras=[camera_1, camera_2, camera_3, camera_4]
    cams={}
    for id, camera in enumerate(cameras): 
        cams[f"Cam{id}"]=camera
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
    "subjectID": "user"}

    with open("utils/sessionMetadata.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

    return f"Rozmary szachownicy: {square_h}x{square_w}, \nRozmiar kwadratu: {square_mm}x{square_mm}"


def record_video(name, del_folder):
    devides=[0,1,2,3]
    # usuń stare foldery
    if del_folder:
        for path in [f"Data/Videos/Cam{id}" for id in devides]:
            if os.path.exists(path):
                shutil.rmtree(path)
    make_dirs()

    caps =[cv2.VideoCapture(file) for file in devides]#index kamery lub sciezka do zdjecia
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Użyj kodeka MJPEG
    image_list=[]
    frame_width_list=[]
    frame_height_list=[]

    image_list=[cap.read() for ind, cap in enumerate(caps)]
    frame_width_list=[int(cap.get(3)) for ind, cap in enumerate(caps)]
    frame_height_list=[int(cap.get(4)) for ind, cap in enumerate(caps)]
    output_files_list = [cv2.VideoWriter(f'Data/Videos/Cam{ind}/{name}_{ind}.mov', fourcc, fps, (frame_width_list[ind], frame_height_list[ind])) for ind, frame in enumerate(image_list)]
    clock_ticks=0
    timer=time.time()
    while True:
        image_list=[cap.read() for ind, cap in enumerate(caps)]
        yield image_list[0][1], image_list[1][1], image_list[2][1], image_list[3][1]
        [output_file.write(image_list[ind][1]) for ind, output_file in enumerate(output_files_list)]
        if time.time()-timer>clock_ticks:
            clock_ticks=clock_ticks+1
        if time.time()-timer>5:
            [cap.release() for cap in caps]
            [out.release() for out in output_files_list]
            cv2.destroyAllWindows()
            break

def calibrate():
    # wczytywanie pliku sessionMetadata.yaml
    sessionMetadata_path="utils/sessionMetadata.yaml"
    myYamlFile = open(sessionMetadata_path)
    sessionMetadata = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    # Kalibracja kazdej kamery
    CamParamDict = {}
    for id, camName in enumerate(sessionMetadata['iphoneModel'].keys()):
        if sessionMetadata['iphoneModel'][camName]!=None:
            # Interinsics 
            permIntrinsicDir = f"CameraIntrinsics/{sessionMetadata['iphoneModel'][camName]}/Deployed/cameraIntrinsics.pickle"
            open_file = open(permIntrinsicDir, "rb")
            CamParams = pickle.load(open_file)
            open_file.close()
            extrinsicPath=f"Data/Videos/{camName}/calibration_{id}.mov"
            CamParams = rotateIntrinsics(CamParams, extrinsicPath)
            CheckerBoardParams = {
                'dimensions': (
                    sessionMetadata['checkerBoard']['black2BlackCornersWidth_n'],
                    sessionMetadata['checkerBoard']['black2BlackCornersHeight_n']),
                'squareSize': 
                    sessionMetadata['checkerBoard']['squareSideLength_mm']}   

            # Extrinsics
            CamParams = calcExtrinsicsFromVideo(
                extrinsicPath,CamParams, CheckerBoardParams, 
                visualize=False, imageUpsampleFactor=4,
                useSecondExtrinsicsSolution = None)

            # zapisz parametry
            if CamParams is not None:
                CamParamDict[camName] = CamParams.copy()
            else:
                CamParamDict[camName] = None
            saveCameraParameters(f"Data/Videos/{camName}/cameraIntrinsicsExtrinsics.pickle", 
                CamParamDict[camName])

def get_netral_pose():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    sessionMetadata_path="utils/sessionMetadata.yaml"
    myYamlFile = open(sessionMetadata_path)
    sessionMetadata = yaml.load(myYamlFile, Loader=yaml.FullLoader)

    keypoints2D={}
    thisConfidence={}
    calibrate_env=True 
    reset_timer=True
    max_val, x_min, y_min, z_min, translation=1, 0, 0, 0, None
    devides=[(id, cam) for id, cam in enumerate(sessionMetadata['iphoneModel'].keys()) if sessionMetadata['iphoneModel'][cam] != None]
    caps =[cv2.VideoCapture(f"Data/Videos/Cam{id}/neutral_pose_{id}.mov") for id, cam in devides]#index kamery lub sciezka do zdjecia
    points2d=[0 for id, camName in devides]
    CameraParamList_selectedCams=[0 for id, camName in devides]
    save={}

    landmarks_on_screen=[False for _, _ in devides]
    break_acc=0
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        static_image_mode=True,
        model_complexity=2) as pose:
        cap=caps[0]
        while cap.isOpened():
            for id, camName in devides:
                cap=caps[id]
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break_acc=break_acc+1
                    break
                else:
                    break_acc=0
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imwrite(f"Data/Videos/{camName}/frame.jpg", image)
                landmarks_on_screen[id] = results.pose_landmarks!=None
                if results.pose_landmarks!=None:
                    image_height, image_width, _ = image.shape
                    #2D keypoints
                    keypoints=sum([[results.pose_landmarks.landmark[ind].x*image_width, results.pose_landmarks.landmark[ind].y*image_width, results.pose_landmarks.landmark[ind].visibility] for ind in range(33)],[])
                    data={"version":1.3,"people":[{"person_id":[-1],"pose_keypoints_2d":keypoints,"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
                    keypoints2D[camName]=np.array(data['people'][0]['pose_keypoints_2d']).reshape(33, 3)[:,:2].reshape(33,1, 2)
                    thisConfidence[camName]=np.array(data['people'][0]['pose_keypoints_2d']).reshape(33, 3)[:,2]

                path=f"Data/Videos/{camName}/cameraIntrinsicsExtrinsics.pickle"
                open_file = open(path, "rb")
                CamParams = pickle.load(open_file)
                open_file.close()
                CameraParamList_selectedCams[id]=CamParams
                ignoreMissingMarkers=False
            if 0 not in landmarks_on_screen:
                points2d=list(keypoints2D.values())
                keypoints3D, confidence3D=np.ones((3,33,1)), np.ones((1,33,1))
                keypoints3D[:,:,0], confidence3D[:,:,0] = triangulateMultiview(CameraParamList_selectedCams, points2d, imageScaleFactor=1, useRotationEuler=False, ignoreMissingMarkers=ignoreMissingMarkers, keypoints2D=[],confidence=list(thisConfidence.values()))
                keypoints3D = return_normalized_keypoints(keypoints3D.tolist(), max_val, x_min, y_min, z_min)
                max_val, x_min, y_min, z_min=return_max_val(keypoints3D)
                keypoints3D, translation = all_transformations(keypoints3D)
                if None not in [max_val, x_min, y_min, z_min, translation[0]]:
                    save={'max_val': max_val, 'x_min, y_min, z_min': (x_min, y_min, z_min),'translation': translation}
            landmarks_on_screen=[False for _, _ in devides]
            if break_acc>10:
                break
    # Sava calibration file
    sessionMetadata.update(save)

    with open('utils/sessionMetadata.yaml', 'w') as file:
        yaml.dump(sessionMetadata, file, default_flow_style=False)

    for cap in caps:
        cap.release()

def run_model():
    # wczytywanie pliku sessionMetadata.yaml
    sessionMetadata_path="utils/sessionMetadata.yaml"
    myYamlFile = open(sessionMetadata_path)
    sessionMetadata = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    devides=[(id, cam) for id, cam in enumerate(sessionMetadata['iphoneModel'].keys()) if sessionMetadata['iphoneModel'][cam] != None]

    # load mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    points2d=[0 for id, camName in devides]
    CameraParamList_selectedCams=[0 for id, camName in devides]
    keypoints2D={}
    thisConfidence={}
    keypoints_2D_list=[0 for id, camName in devides]
    # środek planszy oraz normalizacja x, y, z
    max_val=sessionMetadata["max_val"]
    x_min, y_min, z_min=sessionMetadata["x_min, y_min, z_min"]
    translation=sessionMetadata["translation"]

    landmarks_on_screen=[False for _, _ in devides]

    # caps =[cv2.VideoCapture(f"Data/Videos/Cam{id}/neutral_pose_{id}.mov") for id, cam in devides]#index kamery lub sciezka do zdjecia
    caps =[cv2.VideoCapture(id) for id, cam in devides]#index kamery lub sciezka do zdjecia
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        static_image_mode=True,
        model_complexity=2) as pose:
        cap=caps[0]
        while cap.isOpened():
            if state=="stop":
                break
            for id, camName in devides:
                cap=caps[id]
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    break
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                cv2.imwrite(f"Data/Videos/{camName}/run.jpg", image)
                landmarks_on_screen[id] = results.pose_landmarks!=None
                
                if results.pose_landmarks!=None:
                    image_height, image_width, _ = image.shape
                    #2D keypoints
                    keypoints_2D_list[id]=sum([[results.pose_landmarks.landmark[ind].x*image_width, results.pose_landmarks.landmark[ind].y*image_width] for ind in range(33)],[])
                    keypoints=sum([[results.pose_landmarks.landmark[ind].x*image_width, results.pose_landmarks.landmark[ind].y*image_width, results.pose_landmarks.landmark[ind].visibility] for ind in range(33)],[])
                    data={"version":1.3,"people":[{"person_id":[-1],"pose_keypoints_2d":keypoints,"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
                    keypoints2D[camName]=np.array(data['people'][0]['pose_keypoints_2d']).reshape(33, 3)[:,:2].reshape(33,1, 2)
                    thisConfidence[camName]=np.array(data['people'][0]['pose_keypoints_2d']).reshape(33, 3)[:,2]

                path=f"Data/Videos/{camName}/cameraIntrinsicsExtrinsics.pickle"
                open_file = open(path, "rb")
                CamParams = pickle.load(open_file)
                open_file.close()
                CameraParamList_selectedCams[id]=CamParams
                ignoreMissingMarkers=False
            if 0 not in landmarks_on_screen:
                # 3D kepoints
                points2d=list(keypoints2D.values())
                thisConfidence=list(thisConfidence.values())
                keypoints3D=np.ones((3,33,1))
                confidence3D=np.ones((1,33,1))
                keypoints3D[:,:,0], confidence3D[:,:,0] = triangulateMultiview(CameraParamList_selectedCams, points2d, imageScaleFactor=1, useRotationEuler=False,ignoreMissingMarkers=ignoreMissingMarkers, keypoints2D=[],confidence=thisConfidence)
                # collect data
                keypoints3D=keypoints3D.tolist()
                keypoints3D=return_normalized_keypoints(keypoints3D, max_val, x_min, y_min, z_min)
                keypoints3D = all_transformations(keypoints3D,translation)
                keypoints3D_l=[keypoints3D]+keypoints3D_l[:-1]
                print(keypoints3D_l)
                plot_3d(keypoints3D)
                
                keypoints2D={} #reset directories
                thisConfidence={}

                image_path = "Data/plot.png"  # Podaj ścieżkę do pliku
                plot = mpimg.imread(image_path)

                yield image, plot
            landmarks_on_screen=[False for _, _ in devides]
            X_test = scaler.transform(points_list)
            predictions = model.predict(np.array(X_test))
            predicted_classes = np.argmax(predictions, axis=1)
            print(predicted_classes)




###########
class ImageProcessUI(object):
    def __init__(self, ui_obj):
        self.name = "Image Processor UI"
        self.description = "This class is designed to build UI for our application"
        self.ui_obj = ui_obj

    def create_application_ui(self):
        state = gr.State("Stop")

        with self.ui_obj:
            gr.Markdown("""<div style="text-align: center; font-size: 30px;">Aplikacja</div>""")
            with gr.Tabs():
                with gr.TabItem("Instrukcja"):
                    with gr.Row():
                        gr.Markdown("""# Instrukcja:\n\n
Aplikacja pozwala określić wykonywaną czynność, wymaga ona przygotowania odpowiedniego środowiska oraz kalibracji kamer.\n
### 1. **Kalibracja**
Użyj tej funkcji, aby ustawić parametry kamer, w tym liczbę kamer, rodzaj urządzeń, lokalizacje szachownicy oraz neutralną poze.\n
### 2. **Uruchom**
Po zakończeniu kalibracji możesz uruchomić program, który analizuje obraz z kamer i identyfikuje czynności.\n
Aby zacząć, wybierz odpowiednią opcję z menu.""")  

                with gr.TabItem("Kalibracja"):
                    gr.Markdown("""<div style=" font-size: 18px;">Określenie modeli kamer</div>""")
                    with gr.Row():
                        generate_button = gr.Button("Określ modele kamer")
                        generate_button.click(fn=record_video, inputs=[gr.State("calibration"), gr.State(True)] , outputs=[gr.Image(label="Kamera 1"), gr.Image(label="Kamera 2"), gr.Image(label="Kamera 3"), gr.Image(label="Kamera 4")])

                        #wybieranie modelu telefonu
                        slider = gr.Slider(2, 4, step=1 , value=2, label="Liczba kamer", info="Wybierz liczbę kamer od 2 do 4")
                        folder_contents = os.listdir('CameraIntrinsics')
                        dev_list=[phone for phone in folder_contents if "iPhone" in phone]

                        camera_1 = gr.Dropdown(dev_list, label="Kamera 1", info="Wybierz model kamery")
                        camera_2 = gr.Dropdown(dev_list, label="Kamera 2", info="Wybierz model kamery")
                        camera_3 = gr.Dropdown(dev_list, label="Kamera 3", info="Wybierz model kamery")
                        camera_4 = gr.Dropdown(dev_list, label="Kamera 4", info="Wybierz model kamery")

                    
                    result = gr.Textbox(label="Podsumowanie informacji o aplikacji")
                    generate_button.click(fn=sentence_builder, inputs=[slider, camera_1, camera_2, camera_3, camera_4], outputs=[result])
                    # Szachownica
                    gr.Markdown("""<div style=" font-size: 18px;">Szachownica</div>""")
                    with gr.Row():
                        with gr.Row():
                            record_checkerboard_button = gr.Button("Nagraj szachownice")
                            calibrate_button = gr.Button("Zbieraj dane szachownicy")
                        record_checkerboard_button.click(fn=record_video, inputs=[gr.State("calibration"), gr.State(True)], outputs=[gr.Image(label="Kamera 1"), gr.Image(label="Kamera 2"), gr.Image(label="Kamera 3"), gr.Image(label="Kamera 4")])
                    with gr.Row():
                        square_h = gr.Slider(3, 10, step=1 , value=4, label="Wysokość szachownicy")
                        square_w = gr.Slider(3, 10, step=1 , value=5, label="Szerokość szachownicy")
                        square_mm = gr.Slider(10, 100, step=1 , value=35, label="Rozmiar kwadratu szachownicy [mm]")
                        result = gr.Textbox(label="Parametry kamery")
                        calibrate_button.click(fn=camera_parameters, inputs=[camera_1, camera_2, camera_3, camera_4, square_h, square_w, square_mm], outputs=[result])
                        calibrate_button.click(fn=calibrate)
                        

                    # Neutral pose
                    gr.Markdown("""<div style=" font-size: 18px;">Neutral pose</div>""")
                    with gr.Row():
                        with gr.Row():
                            record_neutralpose_button = gr.Button("Nagraj neutralną pozycję")
                            neutralpose_button = gr.Button("Zbieraj neutralną pozycję")

                        record_neutralpose_button.click(fn=record_video, inputs=[gr.State("neutral_pose"), gr.State(False)], outputs=[gr.Image(label="Kamera 1"), gr.Image(label="Kamera 2"), gr.Image(label="Kamera 3"), gr.Image(label="Kamera 4")])
                        neutralpose_button.click(fn=get_netral_pose)

                    result = gr.Textbox(label="Parametry pozy")
                    # neutralpose_button.click(fn=camera_parameters, outputs=[result])



                with gr.TabItem("Uruchom"):
                    gr.Markdown("""<div style=" font-size: 18px;">Podgląd</div>""")
                    Run = gr.Button("Uruchom detekcje")

                    with gr.Row():
                        # state.change(fn=start_stop, inputs=state)
                        Run.click(fn=start_stop)
                        Run.click(fn=run_model, outputs=[gr.Image(label="Kamera"), gr.Image(label="Triangulation 3D")])
                        Run.click(fn=run_model)

    def launch_ui(self):
        self.ui_obj.launch()

