
# coding: utf-8

# In[2]:


import cv2
import dlib


# In[3]:


ix, iy, ex, ey = -1, -1, -1, -1
cap_from_stream = False
path = "/mnt/24eb92b8-4fc1-4f32-b05c-deaa999ce6cf/Documents/datasets/face/camera_supervisor/1_02_H_082018120000.avi"


# In[4]:


def draw_rec(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    if event == cv2.EVENT_LBUTTONUP:
        ex, ey = x, y
        cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 0)


# In[5]:


def get_crop_size(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if cap_from_stream:
            frame = cv2.resize(frame, (1280, 720))
        cv2.namedWindow('draw_rectangle')
        cv2.setMouseCallback('draw_rectangle', draw_rec, frame)
        print("Choose your area of interest!")
        while 1:
            cv2.imshow('draw_rectangle', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('a'):
                cv2.destroyAllWindows()
                break
        break


# In[6]:


def main():
    get_crop_size(path)
    print(ix, iy, ex, ey)
    cap = cv2.VideoCapture(path)
    print(cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        
        cv2.rectangle(frame, (ix, iy), (ex, ey), (0, 0, 255), 2)
        cv2.imshow("frame", frame)
    
    cv2.destroyAllWindows()


# In[7]:


if __name__ == "__main__":
    main()

