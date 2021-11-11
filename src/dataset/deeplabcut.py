from pathlib import Path
import json

def json_pack(snippets_dir, label, label_index):
    sequence_info = []
    p = Path(snippets_dir)
    for path in p.glob('*.json'):
        json_path = str(path)
        print(path)
        frame_id = int(path.stem.split('_')[-1])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data[0]['people']:
            score , coordinates  = [], []
            skeleton = {}
            keypoints = person['pose_keypoints_3d']
            #print(keypoints)
            for i in range(0, len(keypoints), 3):
                keypoints[i] = float(keypoints[i])
                keypoints[i+1] = float(keypoints[i+1])
                keypoints[i+2] = float(keypoints[i+2])
                if keypoints[i+1] < 3.5 and keypoints[i] > 670:
                    keypoints[i+2] = 0
                coordinates += [float(keypoints[i]), float(keypoints[i + 1])]
                score += [float(keypoints[i + 2])]

            max_x = coordinates[0]
            min_x = coordinates[0]
            max_y = coordinates[1]
            min_y = coordinates[1]
            for j in range(0,len(coordinates),2):
                if coordinates[j] >= max_x and  coordinates[j]<670:
                    max_x = coordinates[j]
                if coordinates[j] <= min_x:
                    min_x = coordinates[j]
                if coordinates[j+1] >= max_y:
                    max_y = coordinates[j+1]
                if coordinates[j+1] <= min_y  and coordinates[j+1]>3.5:
                    min_y = coordinates[j+1]
            frame_x = max_x - min_x
            frame_y = max_y - min_y
            for j in range(0, len(coordinates), 2):
                coordinates[j] = (coordinates[j] - coordinates[8])/frame_x
                coordinates[j+1] = (coordinates[j+1] - coordinates[9])/frame_y #以动物中心点为原点，归一化
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index

    return video_info