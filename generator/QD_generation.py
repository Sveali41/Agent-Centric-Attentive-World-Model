from data.env_dataset_support import *

if __name__ == "__main__":
    rows = 6
    cols = 6
    num_maps = 500
    task_dict = generate_envs_dataset(
                rows, cols, num_maps,
                wall_p_range=(0.1, 0.4),
                door_p_range=(0.0, 0.0),
                key_p_range=(0.0, 0.0),
                max_len=1e7,
                random_gen_max=3e4,
                save_flag= False,
                save_path='/home/siyao/phd_file/Research/rlPractice/MiniGrid/generator/result', start_point_flag=False)

    print("Generated {} maps.".format(len(task_dict)))
    # save the dataset
    save_dic(task_dict, '/home/siyao/phd_file/Research/rlPractice/MiniGrid/generator/data/grid500_6_6.pkl')
