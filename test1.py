from app2 import App2
import time

class TestApp2(App2):
    def __init__(self, use_gpu = True):
        super().__init__(use_gpu)

    def on_log(self, text):
        print("---------on_log", test)


test = TestApp2()

health = test.get_health()
print("---------health:", health)

test.set_config(
        comparison_mode="image_to_folder",
        target_path="D:\\python\\Face Compare\\Face\\Cut\\IMG_0020c.jpg",
        folder_path="D:\\python\\Face Compare\\Face\\Cut",
        threshold=0.6,                        
        result_folder_path="D:\\test-result",         
        batch_processing_enabled=True,        
        batch_size=128,                       
        chunk_size=500,                       
        max_display_results=5                 
)

result = test.start_comparison()

print("---------result:", result)

i = 0
while True:

    status = test.get_status()
    print("---------status:", status)

    i +=1
    if i > 60: break;
    time.sleep(1)

