import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import cosine,cdist

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def load_landmarks(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_landmarks_from_frame(frame, pose):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    # image.flags.writeable = True
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return [{
            "id": i,
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility
        } for i, landmark in enumerate(landmarks)]
    return None


def calculate_landmark_distances(landmarks1, landmarks2):
    distances = []
    for lm1, lm2 in zip(landmarks1, landmarks2):
        sum=[]
        for i,j in zip(lm1,lm2):
            distance = np.sqrt((i['x'] - j['x']) ** 2 + (i['y'] - j['y']) ** 2 + (i['z'] - j['z']) ** 2)
            sum.append(distance)
        distances.append(np.mean(sum))
    #     print(lm1)
    return distances



def interpolate_landmarks(real_time_landmarks, target_frame_count):
    original_frame_count = len(real_time_landmarks)
    # print("Original FrameCount",original_frame_count)
    frame_indices = np.arange(original_frame_count)
    # print(frame_indices)
    target_indices = np.linspace(0, original_frame_count - 1, target_frame_count)
    # print(target_indices)
    
    interpolated_landmarks = []
    for i in range(33):  # Number of landmarks
        xs = [frame['landmarks'][i]['x'] for frame in real_time_landmarks]
        # print(xs)
        ys = [frame['landmarks'][i]['y'] for frame in real_time_landmarks]
        zs = [frame['landmarks'][i]['z'] for frame in real_time_landmarks]
        
        interp_x = interp1d(frame_indices, xs, kind='cubic')(target_indices)
        interp_y = interp1d(frame_indices, ys, kind='cubic')(target_indices)
        interp_z = interp1d(frame_indices, zs, kind='cubic')(target_indices)
        for j, idx in enumerate(target_indices):
            if len(interpolated_landmarks) <= j:
                interpolated_landmarks.append({
                    "frame": j,
                    "landmarks": []
                })
            interpolated_landmarks[j]['landmarks'].append({
                "id": i,
                "x": interp_x[j],
                "y": interp_y[j],
                "z": interp_z[j],
                "visibility": 1.0  # Assuming visibility is 1.0 for interpolated landmarks
            })
    
    return interpolated_landmarks
def calculate_cosine_distances(landmarks1, landmarks2):
    distances = []
    for lm1, lm2 in zip(landmarks1, landmarks2):
        sum_cosine = []
        for i, j in zip(lm1, lm2):
            vector1 = np.array([i['x'], i['y'], i['z']])
            vector2 = np.array([j['x'], j['y'], j['z']])
            cosine_distance = 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            sum_cosine.append(cosine_distance)
        distances.append(np.mean(sum_cosine))
    return distances

def cosine_distance_3d(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    cosine_similarity = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    return 1 - cosine_similarity

print("Choose on of the following videos:\n1) Tornado Kick\n2) Simple Kick")
choice=int(input("Your input:"))
if choice==1:
    path="tornado_kick_demo.mp4"
    video_path="tornado_kick_demo.mp4"
else:
    video_path="basic_karatee.mp4"
    path="basic_karatee.mp4"

# Load the pre-recorded landmarks
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Reference Video', cv2.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Reference Video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('Reference Video')
# for i in reference_landmarks:
#     print(i['frame'])
 

cap=cv2.VideoCapture(path)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_number=0
    landmarks_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        # Breaks loop when there are no more frames to read
        if not ret:
            break
        
        
        
        # Coloring to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(results)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Save the landmarks data to the list
            landmarks_data.append({
                "frame": frame_number,
                "landmarks": [{
            "id": i,
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility
        } for i, landmark in enumerate(landmarks)]
            })
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Reference video', image)
        frame_number += 1

    
        
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # Save the landmarks data to a pickle file
    with open('landmarks_reference.pkl', 'wb') as file:
        pickle.dump(landmarks_data, file)

reference_landmarks = load_landmarks('landmarks_reference.pkl')


# Calculate the frame rate of the reference video
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("Error: Could not open video.")

# Get the frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# Get the total number of frames
frame_count = len(reference_landmarks)

# Calculate duration in seconds
video_duration = frame_count / fps
print(video_duration)

cap.release()  # seconds
# reference_frame_count = len(reference_landmarks)


# Start capturing real-time footage
frame_number = 0
real_time_landmarks = []
cap_realtime = cv2.VideoCapture(0)
print(cap_realtime.get(cv2.CAP_PROP_FPS))
for i in range(3, 0, -1):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # create a black frame
    cv2.putText(frame, str(i), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
    cv2.imshow('Real-Time Footage', frame)
    cv2.waitKey(1)
    time.sleep(1)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    start_time = time.time()
    

    while cap_realtime.isOpened() and (time.time() - start_time) < video_duration:
        ret, frame = cap_realtime.read()
        if not ret:
            break
        
        landmarks = extract_landmarks_from_frame(frame, pose)
        if landmarks:
            real_time_landmarks.append({
                "frame": frame_number,
                "landmarks": landmarks
            })
            frame_number += 1

            # Draw landmarks on the real-time frame
            mp_drawing.draw_landmarks(frame, pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the real-time frame
        cv2.imshow('Real-Time Footage', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Time is", time.time()-start_time)
    cap_realtime.release()
    cv2.destroyAllWindows()
    ref_landmarks=[]
    realtime_landmarks=[]
    # for l in real_time_landmarks:
    #     for i in l['landmarks']:
    #         print(i['id'])
    if len(real_time_landmarks) < frame_count:
        print(f"Real-time footage has {len(real_time_landmarks)} frames; reference has {frame_count} frames.")
        real_time_landmarks = interpolate_landmarks(real_time_landmarks, frame_count)
    # for i in real_time_landmarks:
    #     print(i)
    for i in range(frame_count):
        ref_landmarks.append(reference_landmarks[i]['landmarks'])
        realtime_landmarks.append(real_time_landmarks[i]['landmarks'])
    print(len(ref_landmarks), len(real_time_landmarks))
    distances = calculate_landmark_distances(realtime_landmarks, ref_landmarks)
    avg_distance = np.mean(distances)
    avg_cosine_distance=[]
    # for i in range(frame_count):
    #     rt = np.array([[realtime_landmarks[i][j]['x'],realtime_landmarks[i][j]['y'],realtime_landmarks[i][j]['z']] for j in range(33)])
    #     ref= np.array([[ref_landmarks[i][j]['x'],ref_landmarks[i][j]['y'],ref_landmarks[i][j]['z']] for j in range(33)])
    #     cosine_distances=fastdtw(rt,ref,dist=cosine)[0]
    #     avg_cosine_distance.append(np.mean(cosine_distances))
    # print(avg_cosine_distance)
    for i in range(frame_count):
        rt = np.array([np.array([realtime_landmarks[i][j]['x'],realtime_landmarks[i][j]['y'],realtime_landmarks[i][j]['z']]) for j in range(33)])
        ref= np.array([np.array([ref_landmarks[i][j]['x'],ref_landmarks[i][j]['y'],ref_landmarks[i][j]['z']]) for j in range(33)])
        
        cosine_distances=cdist(rt,ref,'cosine')
        # fastdtw(rt,ref,dist=cosine)[0]
        avg_cosine_distance.append(np.mean(cosine_distances))
    # print(avg_cosine_distance)
    a=sum(1 for D in avg_cosine_distance if D<0.2)
    acc=a/frame_count
    print(f"Average Accuracy: {acc*100} %")
    # a=sum(1 for D in avg_cosine_distance if D<0.3)
    # acc=a/frame_count
    # print(f"Frame {i}: Average distance: {acc*100}")
    # print([[realtime_landmarks[1][j]['x'],realtime_landmarks[1][j]['y'],realtime_landmarks[1][j]['z']] for j in range(33)])