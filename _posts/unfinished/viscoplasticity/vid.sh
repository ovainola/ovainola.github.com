ffmpeg -framerate 24 -i %03d_plot.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
