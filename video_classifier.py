from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import scipy.misc
import skvideo.io
import json
import hashlib
from werkzeug.contrib.cache import MemcachedCache
cache = MemcachedCache(['127.0.0.1:11211'])


def get_hash(video):
    hash = ""
    with open(video,'rb') as f:
        sha1obj = hashlib.sha1()
        sha1obj.update(f.read())
        hash = sha1obj.hexdigest()
    return hash
        

def analyze_video(video):
    hash = get_hash(video)
    stats = cache.get(hash)
    if stats:
        print "hash hit"
        return stats
    else:
        print "hash missed"
        model = ResNet50(weights='imagenet')
        vd = skvideo.io.vread(video)
        metadata = skvideo.io.ffprobe(video)
        exec("frame_rate = {}".format(metadata["video"]["@r_frame_rate"]))
        frames = vd.shape[0]
        stats = []
        for i in range(frames):
            x = vd[i,:,:,:]
            x = scipy.misc.imresize(x, (224, 224))
            x = x.astype(np.float32)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
            _, obj, prob = decode_predictions(preds, top=1)[0][0]
            stats.append({
                "frame": i,
                "obj": obj,
                "prob": prob,
                "second": i / frame_rate,
            })
        # never expire
        cache.set(hash, stats, timeout=0)
        return stats
        
def print_summary(stats, threshold=.2):
    out = get_summary(stats, threshold)
    for obj, seconds in out.iteritems():
        seconds = sorted(list(set(seconds)))
        seconds = [str(s) for s in seconds]
        print "{} appears at second {}".format(obj, ", ".join(seconds))
        
def get_summary(stats, threshold=.2):
    out = {}
    for frame_stat in stats:
        if frame_stat["prob"] < threshold:
            continue
        if frame_stat["obj"] not in out:
            out[frame_stat["obj"]] = []
        if frame_stat["second"] not in out[frame_stat["obj"]]:
            out[frame_stat["obj"]].append(frame_stat["second"])
    return out

if __name__ == "__main__":
    stats = analyze_video("giphy.mp4")
    print_summary(stats)
