from webcam import VideoPlayer
def main(file_name):
    input_file = f'depth_image_data/{file_name}'
    output_file = r'depth_image_data\\jason'
    load = VideoPlayer(input_file)
    load.play(output_file)

if __name__ == '__main__':
    main('001.mp4')