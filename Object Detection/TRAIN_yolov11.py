from ultralytics import YOLO, checks, hub
checks()

if __name__ == '__main__':
    #yolov11n
    hub.login('f5c925aa8e5de707c0162b82ee9609209ab6b58c0d')
    model = YOLO('https://hub.ultralytics.com/models/t0Ws0dB8UAUEvEgO237u')
    
    #yolov11x
    #hub.login('f5c925aa8e5de707c0162b82ee9609209ab6b58c0d')
    #model = YOLO('https://hub.ultralytics.com/models/u1g0Uqss5hvNkiBCuKSX')

    results = model.train()
