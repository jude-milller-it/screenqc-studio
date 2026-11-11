"""
SegFormer prediction entry point.

Modes:
  'predict'       — single image
  'batch_compare' — batch predict multiple images to compare with DeepLabV3 results
  'dir_predict'   — predict all images in a folder
  'fps'           — measure FPS
  'video'         — webcam/video
"""
import os
import time
import cv2
import numpy as np
from PIL import Image

from segformer_infer import SegFormerInference


if __name__ == "__main__":
    # ====================== 配置 ======================
    # SegFormer 变体和权重路径
    variant = "mit_b2"
    model_path = f"logs_segformer_{variant}/best_epoch_weights.pth"

    # 如果还没训练好，可以 fallback 到其他权重
    if not os.path.exists(model_path):
        print(f"[WARNING] {model_path} not found, trying last_epoch_weights...")
        model_path = f"logs_segformer_{variant}/last_epoch_weights.pth"

    # 实例化推理器
    segformer = SegFormerInference(
        num_classes=4,
        variant=variant,
        model_path=model_path,
        input_shape=[512, 512],
        use_postprocess=True,
        use_tta=True,
        cuda=True,
    )

    # ---- 模式选择 ----
    mode = "batch_compare"
    # -------------------------------------------------------
    count = True
    name_classes = ["background", "oil", "stain", "scratch"]

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = segformer.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "batch_compare":
        # 批量预测，与 DeepLabV3 结果并列对比
        img_list = ["img/Oil_0125.jpg", "img/Scr_0368.jpg", "img/Sta_0217.jpg"]
        save_dir = "out_segformer"
        os.makedirs(save_dir, exist_ok=True)

        for img_path in img_list:
            if not os.path.exists(img_path):
                print(f"跳过：{img_path}")
                continue
            img_name = os.path.basename(img_path)
            name, ext = os.path.splitext(img_name)
            image = Image.open(img_path)
            res = segformer.detect_image(image, count=count, name_classes=name_classes)
            save_path = os.path.join(save_dir, f"{name}_{variant}{ext}")
            res.save(save_path)
            print(f"已保存 {save_path}")
        print(f"\n全部完成，结果在 {save_dir}/ 文件夹里")

    elif mode == "dir_predict":
        from tqdm import tqdm
        dir_origin_path = "img/"
        dir_save_path = "img_out_segformer/"
        img_names = os.listdir(dir_origin_path)
        os.makedirs(dir_save_path, exist_ok=True)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg',
                                          '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = segformer.detect_image(image)
                r_image.save(os.path.join(dir_save_path, img_name))
        print(f"全部完成，结果在 {dir_save_path}/")

    elif mode == "fps":
        fps_image_path = "img/Oil_0001.jpg" if os.path.exists("img/Oil_0001.jpg") else "img/Oil_0125.jpg"
        img = Image.open(fps_image_path)
        test_interval = 100
        tact_time = segformer.get_FPS(img, test_interval)
        print(f'{tact_time:.4f} seconds, {1/tact_time:.2f} FPS, @batch_size 1')

    elif mode == "video":
        capture = cv2.VideoCapture(0)
        while True:
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(segformer.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("SegFormer Video", frame)
            if cv2.waitKey(1) & 0xff == 27:
                break
        capture.release()
        cv2.destroyAllWindows()

    else:
        raise AssertionError(f"Unknown mode: {mode}")
