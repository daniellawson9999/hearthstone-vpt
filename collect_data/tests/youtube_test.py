from pytube import YouTube

#video_link = 'https://www.youtube.com/watch?v=35M2m4_0ZHk'
video_link = 'http://youtube.com/watch?v=2lAe1cqCOXo'
yt = YouTube(video_link)

yt.streams.filter(file_extension='mp4', res='360p')[-1].download()