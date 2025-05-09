import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("/home/rcasal/ros2_ws/src/nanoowl/data/owl_image_encoder_patch32.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:

    engine = runtime.deserialize_cuda_engine(f.read())
    if engine:
        print("✅ Motor cargado correctamente en TensorRT")
    else:
        print("❌ Error al cargar el motor")
