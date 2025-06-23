# main.py

import os
import time
import shutil
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # این خط دیگر نیازی نیست، چون ایمپورت ها از src. آغاز می شوند

from src.face_recognition_app import FaceRecognitionApp
from config import (
    STATUS_COMPLETE, STATUS_ERROR, STATUS_PROCESSING, STATUS_WAITING, STATUS_PAUSED,
    # FIX: STATUS_INCOMPLETE را ایمپورت می کنیم
    STATUS_INCOMPLETE,
    BASE_PROJECT_DIR, RESULTS_FOLDER, SORTING_OUTPUT_FOLDER, DATABASE_FOLDER, LOGS_FOLDER
)

# --- مسیر مطلق به فایل dummy.jpg شما ---
# FIX: نام فایل dummy.jpg را به dummy_face.jpg تغییر دهید و مطمئن شوید که این فایل حاوی یک چهره باشد.
SOURCE_DUMMY_JPG = os.path.join(BASE_PROJECT_DIR, "dummy_face.jpg")  # <-- تغییر نام به dummy_face.jpg


def create_dummy_file(filepath: str):
    """
    تابع کمکی برای ایجاد فایل های تصویری dummy برای تست با کپی کردن یک فایل JPG معتبر.
    **این فایل باید حاوی یک چهره باشد تا تشخیص چهره کار کند.**
    اگر فایل SOURCE_DUMMY_JPG پیدا نشود، یک خطا ایجاد می کند.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        if os.path.exists(SOURCE_DUMMY_JPG):
            shutil.copy2(SOURCE_DUMMY_JPG, filepath)
            print(f"Dummy file created by copying '{os.path.basename(SOURCE_DUMMY_JPG)}' to: '{filepath}'")
        else:
            raise FileNotFoundError(
                f"Source dummy_face.jpg not found at '{SOURCE_DUMMY_JPG}'. Please place a valid JPG file with a face in your project root.")
    else:
        print(f"Dummy file already exists: '{filepath}'")


# --- بلوک اجرای اصلی ---
if __name__ == "__main__":
    # اطمینان از وجود تمام پوشه های لازم
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(SORTING_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
    os.makedirs(LOGS_FOLDER, exist_ok=True)

    # تعریف مسیرها برای فایل های تصویری dummy
    dummy_target_img_path = os.path.join(BASE_PROJECT_DIR, "test_target.jpg")
    dummy_comparison_folder = os.path.join(BASE_PROJECT_DIR, "comparison_folder")
    dummy_target_folder = os.path.join(BASE_PROJECT_DIR, "target_folder")

    dummy_folder_img1_path = os.path.join(dummy_comparison_folder, "test_folder_1.jpg")
    dummy_folder_img2_path = os.path.join(dummy_comparison_folder, "test_folder_2.png")
    dummy_folder_img3_path = os.path.join(dummy_comparison_folder, "test_folder_3.jpg")
    dummy_target_folder_img1_path = os.path.join(dummy_target_folder, "test_target_folder_1.jpg")
    dummy_target_folder_img2_path = os.path.join(dummy_target_folder, "test_target_folder_2.jpg")

    # ایجاد تمام فایل های dummy
    create_dummy_file(dummy_target_img_path)
    create_dummy_file(dummy_folder_img1_path)
    create_dummy_file(dummy_folder_img2_path)
    create_dummy_file(dummy_folder_img3_path)
    create_dummy_file(dummy_target_folder_img1_path)
    create_dummy_file(dummy_target_folder_img2_path)

    app_instance = FaceRecognitionApp(use_gpu=True)  # استفاده از GPU به صورت پیش فرض اگر در دسترس است

    print("\n--- Test 1: Image to Folder Comparison ---")
    app_instance.config(
        comparison_mode="image_to_folder",
        target_path=dummy_target_img_path,
        folder_path=dummy_comparison_folder,
        threshold=0.6,
        batch_processing_enabled=True,
        batch_size=128,
        max_display_results=5,
        enable_face_landmarks=True,
        enable_age_gender_detection=True
    )
    if app_instance.compare():
        print("Image to Folder comparison started. Monitoring status...")
        while app_instance.status()['status'] in [STATUS_WAITING, STATUS_PROCESSING, STATUS_PAUSED]:
            current_status = app_instance.status()
            print(f"Status: {current_status['status']} ({current_status['progress']}%) - {current_status['details']}")
            if current_status['progress'] > 20 and current_status[
                'status'] == STATUS_PROCESSING and not app_instance.pause_requested:
                print("--- Requesting PAUSE ---")
                app_instance.pause()
            if current_status['status'] == STATUS_PAUSED:
                print("--- Paused. Waiting 3 seconds then RESUMING ---")
                time.sleep(3)
                app_instance.resume()
            time.sleep(1)
        final_status = app_instance.status()
        print(
            f"Final status: {final_status['status']} - {final_status['details']} (Progress: {final_status['progress']}%)")
    else:
        print("Failed to start Image to Folder comparison.")

    print("\n--- Test 2: Folder to Folder Comparison ---")
    app_instance.config(
        comparison_mode="folder_to_folder",
        target_folder_path=dummy_target_folder,
        folder_path=dummy_comparison_folder,
        threshold=0.65,
        batch_size=64,
        enable_face_landmarks=False,
        enable_age_gender_detection=False
    )
    if app_instance.compare():
        print("Folder to Folder comparison started. Monitoring status...")
        while app_instance.status()['status'] in [STATUS_WAITING, STATUS_PROCESSING, STATUS_PAUSED]:
            current_status = app_instance.status()
            print(f"Status: {current_status['status']} ({current_status['progress']}%) - {current_status['details']}")
            time.sleep(1)
        final_status = app_instance.status()
        print(
            f"Final status: {final_status['status']} - {final_status['details']} (Progress: {final_status['progress']}%)")
    else:
        print("Failed to start Folder to Folder comparison.")

    print("\n--- Test 3: Sorting Faces ---")
    app_instance.config(
        comparison_mode="sorting",
        folder_path=dummy_comparison_folder,
        sorting_output_path=SORTING_OUTPUT_FOLDER,
        similarity_threshold_sorting=0.8,
        min_faces_per_group=1,
        merge_folders_enabled=True,
        enable_face_landmarks=True,
        enable_quality_enhancement=True
    )
    if app_instance.compare():
        print("Sorting faces started. Monitoring status...")
        while app_instance.status()['status'] in [STATUS_WAITING, STATUS_PROCESSING, STATUS_PAUSED]:
            current_status = app_instance.status()
            print(f"Status: {current_status['status']} ({current_status['progress']}%) - {current_status['details']}")
            time.sleep(1)
        final_status = app_instance.status()
        print(
            f"Final status: {final_status['status']} - {final_status['details']} (Progress: {final_status['progress']}%)")
    else:
        print("Failed to start Sorting faces.")

    print("\n--- Test 4: Stop Operation ---")
    app_instance.config(
        comparison_mode="image_to_folder",
        target_path=dummy_target_img_path,
        folder_path=dummy_comparison_folder,
        batch_size=1,
    )
    if app_instance.compare():
        print("Slow operation started. Waiting 2 seconds then requesting stop...")
        time.sleep(2)
        if app_instance.stop():
            print("Stop request sent. Monitoring status until completion...")
            while app_instance.status()['status'] in [STATUS_WAITING, STATUS_PROCESSING, STATUS_PAUSED]:
                current_status = app_instance.status()
                print(
                    f"Status: {current_status['status']} ({current_status['progress']}%) - {current_status['details']}")
                time.sleep(0.5)
            final_status = app_instance.status()
            print(
                f"Final status after stop: {final_status['status']} - {final_status['details']} (Progress: {final_status['progress']}%)")
            if final_status['status'] == STATUS_INCOMPLETE:
                print("Operation successfully cancelled.")
            else:
                print("Operation did not cancel as expected or completed before cancellation.")
        else:
            print("Failed to send stop command.")
    else:
        print("Failed to start operation for stop test.")

    print("\n--- Final App Status and Health Check ---")
    print(app_instance.status())
    print(app_instance.health())

    app_instance.db_manager.close()