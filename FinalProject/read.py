import cv2
import os
import shutil

def readVideo(video_path):
    video = cv2.VideoCapture(os.path.join(video_path, 'example.mp4'))

    cnt = 0
    store_path = os.path.join(video_path, "output/")
    if os.path.exists(store_path):
        print(f'[INFO] Delete old folder')
        # os.remove(store_path)
        shutil.rmtree(store_path)
    os.makedirs(store_path)

    while True:
        # 读取下一帧
        success, frame = video.read()
        if not success:
            break
        # 显示帧
        # cv2.imshow("Frame", frame)
        cv2.imwrite(store_path+str(cnt)+".jpg", frame)
        cnt += 1
        # 等待按键按下
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放资源
    video.release()
    # cv2.destroyAllWindows()
    print(f'[INFO] Total images extracted from video is: {cnt}')