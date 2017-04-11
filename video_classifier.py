from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import scipy.misc
import skvideo.io
import json
import hashlib
from werkzeug.contrib.cache import MemcachedCache
import tensorflow as tf
cache = MemcachedCache(['127.0.0.1:11211'])

imagenet_model = ResNet50(weights='imagenet')
graph = tf.get_default_graph()


def get_hash(video):
    hash = ""
    with open(video,'rb') as f:
        sha1obj = hashlib.sha1()
        sha1obj.update(f.read())
        hash = sha1obj.hexdigest()
    return hash
        

def analyze_video(video, threshold=.2):
    hash = get_hash(video)
    stats = cache.get(hash)
    if stats:
        print "hash hit"
        return stats
    else:
        print "hash missed"
        global graph
        with graph.as_default():
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

                preds = imagenet_model.predict(x)
                for pred in decode_predictions(preds, top=5)[0]:
                    _, obj, prob = pred
                    if prob >= threshold:
                        stats.append({
                            "frame": i,
                            "obj": obj.replace("_", " "),
                            "prob": prob,
                            "second": i / frame_rate,
                        })
            stats.sort(key=lambda x:x["prob"], reverse=True)
            # never expire
        cache.set(hash, stats, timeout=0)
        return stats
        
def print_summary(stats):
    out = get_summary(stats)
    for obj, seconds in out.iteritems():
        seconds = sorted(list(set(seconds)))
        seconds = [str(s) for s in seconds]
        print "{} appears at second {}".format(obj, ", ".join(seconds))
        
def get_summary(stats):
    out = {}
    for frame_stat in stats:
        if frame_stat["obj"] not in out:
            out[frame_stat["obj"]] = []
        if frame_stat["frame"] not in out[frame_stat["obj"]]:
            out[frame_stat["obj"]].append(frame_stat["frame"])
    return out

if __name__ == "__main__":
    stats = analyze_video("giphy.mp4")
    print_summary(stats)
