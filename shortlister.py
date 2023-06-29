import time
import utils

selected_profile = utils.select_profile()
root_path = f"./Data/Profiles/{selected_profile}"


def main():
    start = time.perf_counter()
    utils.shortlist_resumes(root_path)
    end = time.perf_counter()
    print(f'\nFinished in {round(end-start, 2)} seconds')


if __name__ == "__main__":
    main()
