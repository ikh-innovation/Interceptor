from roboflow import Roboflow

rf = Roboflow(api_key="E4P1asOlS3HJ2EwCHUOy")
project = rf.workspace("drone-p4ro3").project("lrddv2-fhowg")
version = project.version(4)
dataset = version.download("yolov8")
                