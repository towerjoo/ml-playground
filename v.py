import skvideo.io
import scipy.misc
import json

#vd = skvideo.io.vread("giphy.mp4")
metadata = skvideo.io.ffprobe("giphy.mp4")
print(json.dumps(metadata["video"], indent=4))
#first_frame = vd[0,:,:,:]
#first_frame = scipy.misc.imresize(first_frame, (224, 224))
#scipy.misc.imsave("first2.png", first_frame)

