import os
import shutil

def organize_ravdess_files():
    """
    Organizes RAVDESS audio and video files from source directories into
    a new structure sorted by actor.
    """
    
    # --- Configuration ---
    # Base path for video source folders (e.g., "...\Video_Speech_Actor_")
    video_source_base = r"D:\Yeskendir_files\ravdess\Video_Speech_Actor_"
    
    # Path to the single folder containing all audio files
    audio_source_dir = r"D:\Yeskendir_files\ravdess\Audio_Speech_Actors_01-24"
    
    # The new base directory where you want the "RAVDESS" folder to be created
    target_base_dir = r"D:\Yeskendir_files\RAVDESS_1"
    
    # The name of the main organized folder
    target_ravdess_dir = os.path.join(target_base_dir, "RAVDESS")
    # --- End Configuration ---

    print(f"Starting file organization...")
    print(f"Target directory will be: {target_ravdess_dir}")

    # # Create the main target directory (e.g., ...\ravdes\RAVDESS)
    # try:
    #     os.makedirs(target_ravdess_dir, exist_ok=True)
    # except OSError as e:
    #     print(f"Error creating base directory {target_ravdess_dir}: {e}")
    #     return

    # # Process Video Files first
    # print("\n--- Processing Video Files ---")
    # for i in range(1, 25):  # Loop for Actor 01 to 24
    #     actor_num_str = str(i).zfill(2)  # "01", "02", ..., "24"
    #     actor_name = f"ACTOR{actor_num_str}"
        
    #     # 1. Create the target actor folder (e.g., ...\RAVDESS\ACTOR01)
    #     actor_target_dir = os.path.join(target_ravdess_dir, actor_name)
    #     os.makedirs(actor_target_dir, exist_ok=True)
        
    #     # 2. Define the source video folder for this actor
    #     actor_video_source_dir = f"{video_source_base}{actor_num_str}"
        
    #     # Account for the nested "Actor_XX" folder as per user clarification
    #     nested_actor_folder = f"Actor_{actor_num_str}"
    #     actor_video_source_dir_nested = os.path.join(actor_video_source_dir, nested_actor_folder)
        
    #     # 3. Check if the source video folder exists
    #     if not os.path.isdir(actor_video_source_dir_nested):
    #         print(f"[Warning] Video source folder not found: {actor_video_source_dir_nested}")
    #         continue
            
    #     print(f"Copying video files for {actor_name} from {actor_video_source_dir_nested}...")
        
    #     # 4. Copy all .mp4 files
    #     for filename in os.listdir(actor_video_source_dir_nested):
    #         if filename.endswith(".mp4"):
    #             source_file_path = os.path.join(actor_video_source_dir_nested, filename)
    #             target_file_path = os.path.join(actor_target_dir, filename)
                
    #             # Copy the file
    #             try:
    #                 if not os.path.exists(target_file_path):
    #                     shutil.copy2(source_file_path, target_file_path)
    #                 else:
    #                     print(f"  - Skip: {filename} (already exists)")
    #             except Exception as e:
    #                 print(f"  [Error] Failed to copy {filename}: {e}")

    # # Process Audio Files
    print("\n--- Processing Audio Files ---")
    if not os.path.isdir(audio_source_dir):
        print(f"[Error] Audio source folder not found: {audio_source_dir}")
        print("Audio file processing skipped.")
        return

    print(f"Scanning {audio_source_dir} for nested actor folders...")
    audio_files_copied = 0
    audio_files_skipped = 0
    
    # Loop for Actor 01 to 24
    for i in range(1, 25):
        actor_num_str = str(i).zfill(2)  # "01", "02", ..., "24"
        actor_name = f"ACTOR{actor_num_str}"
        
        # 1. Define the source audio folder for this actor (e.g., ...\Audio_Speech_Actors_01-24\Actor_01)
        nested_actor_folder = f"Actor_{actor_num_str}"
        actor_audio_source_dir = os.path.join(audio_source_dir, nested_actor_folder)
        
        # 2. Define the target directory (e.g., ...\RAVDESS\ACTOR01)
        # This folder should already exist from the video processing step
        actor_target_dir = os.path.join(target_ravdess_dir, actor_name)
        
        if not os.path.isdir(actor_target_dir):
             print(f"  [Warning] Target directory {actor_target_dir} not found. Creating it.")
             os.makedirs(actor_target_dir, exist_ok=True)
             
        # 3. Check if the source audio folder exists
        if not os.path.isdir(actor_audio_source_dir):
            print(f"  [Warning] Audio source folder not found: {actor_audio_source_dir}")
            continue
            
        print(f"  Copying audio files for {actor_name} from {actor_audio_source_dir}...")
        
        # 4. Loop through all files in this actor's audio source folder
        for filename in os.listdir(actor_audio_source_dir):
            if filename.endswith(".wav"):
                try:
                    source_file_path = os.path.join(actor_audio_source_dir, filename)
                    target_file_path = os.path.join(actor_target_dir, filename)
                    
                    if not os.path.exists(target_file_path):
                        shutil.copy2(source_file_path, target_file_path)
                        audio_files_copied += 1
                    else:
                        audio_files_skipped += 1
                except Exception as e:
                    print(f"    [Error] Failed to copy {filename}: {e}")

    print(f"Audio file copy complete. Copied: {audio_files_copied}, Skipped: {audio_files_skipped}")
if __name__ == "__main__":
    organize_ravdess_files()