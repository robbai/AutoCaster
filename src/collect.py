from typing import List
import os
import sys

from tqdm import tqdm

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ytsubtitles")
sys.path.insert(0, path)

import ytsubtitles.subtitles_downloader as ytsubtitles


def get_subtitles(video_id: str) -> List[str]:
    try:
        video_info = ytsubtitles.__get_video_info__(video_id)
        track_urls = ytsubtitles.__get_sub_track_urls__(video_info)
        target_track_url = ytsubtitles.__select_target_language_track_url(
            track_urls, "en"
        )
        subs_data = ytsubtitles.__get_subs_data__(target_track_url)
        srt = ytsubtitles.to_srt(subs_data)

        return [
            line.strip().capitalize()
            for line in srt.replace("&#39;", "'").split("\n")[2::4]
        ]
    except Exception:
        # print(video_id + ": " + (str(e) if str(e) else "Error"))
        return []


if __name__ == "__main__":
    subtitles = []
    for video_id in tqdm(open("ids.txt", "r").readlines()):
        subtitles += get_subtitles(video_id.strip())
    f = open("data.txt", "w")
    f.write(" ".join(subtitles))
    f.close()
