from pathlib import Path

path = Path("/mnt/data/cyh/VLA-long-tail/vla_scripts/parallel_libero_evaluator_egl.py")
text = path.read_text()

old = '''        video_save_dir = os.path.join(self.save_dir, f'{task_id}_{task_description}')
        os.makedirs(video_save_dir, exist_ok=True)
        write_video(replay_images, os.path.join(video_save_dir, f'episode{episode}_success={success}.gif'), 
                    texts=None, fps=self.cfg.fps)
        
        self.logger.info(f'Task {task_id} {task_description} episode {episode}: success {success}')
'''

new = '''        # Skip GIF writing on headless servers without ffmpeg; metrics only need success summaries.
        self.logger.info(f'Task {task_id} {task_description} episode {episode}: success {success}')
'''

if new not in text:
    if old not in text:
        raise SystemExit("video block not found")
    text = text.replace(old, new, 1)
    path.write_text(text)

print("patched")
