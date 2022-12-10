
import ffmpeg
import os


def extract(
    file,
    fps,
    output_dir= "frames"
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        stream = (
            ffmpeg.input(file.name)
            .filter("fps", fps=str(fps))
            .output(f"{output_dir}/test-%d.jpg", start_number=0)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
        raise e
