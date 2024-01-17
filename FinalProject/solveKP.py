import cv2

def getKeypoints(first_image_path):
    first_image = cv2.imread(first_image_path)
    
    cv2.namedWindow("image")
    kp = []
    
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            print(f'[INFO] Your select keypoint: ({x}, {y})')
            kp.append((x, y))
            cv2.circle(first_image, (x, y), 2, (0, 0, 255))
            cv2.putText(first_image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,255))
            cv2.imshow("image", first_image)

    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while(1):
        cv2.imshow("image", first_image)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    kp = sorted(kp, key=lambda point: (point[0], point[1]))
    print(kp)
    print(f'[INFO] You have chosen {len(kp)} keypoints.')