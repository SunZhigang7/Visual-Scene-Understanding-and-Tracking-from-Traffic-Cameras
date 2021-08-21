import requests


def get_videos(video_number):
    print('Beginning file download with requests')
    i = 1250
    notfound = 0
    number = 1

    while number <= video_number and notfound <= 2000:
        url = 'https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.' + "{:05d}".format(i) + '.mp4'
        # print(url)
        # print('looking for 00001.' + "{:05d}".format(i) + '.mp4')
        r = requests.get(url)
        if r.status_code == 404:
            notfound = notfound + 1
        else:
            with open('data/all_london_traffic_videos/video' + str(number) + '.mp4', 'wb') as f:
                f.write(r.content)

            # Retrieve HTTP meta-data
            # print(r.status_code)
            # print(r.headers['content-type'])
            # print(r.encoding)
            print('Getting ' + str(number) + ' video(s)')
            number = number + 1
            notfound = 1
        i = i + 1

