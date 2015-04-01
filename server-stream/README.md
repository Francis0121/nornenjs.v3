# Nornenjs

We were implemented based on ["node-cuda"](https://github.com/kashif/node-cuda).

We added some library "node-cuda" `cuda_runtime_api.h` and `helper_math.h`. And change some code that implement receive multi parameter to CUDA source.

And we used ["BinaryJS"](http://binaryjs.com/) and ["socket.io"](http://socket.io/).

## Prerequisites

You will need to have a CUDA compatible GPU as well as the latest [CUDA Drivers and Toolkit](https://developer.nvidia.com/cuda-downloads) installed for your platform.
 
Currently only tested on Ubuntu 12.04.14 32bit.

And you will need to jpeg, png library. If you use ubuntu, you install library `libjpeg-dev` and `libpng-dev`.

```
sudo apt-get install libjpeg-dev
sudo apt-get install libpng-dev
```


## Installation

To obtain and build the bindings:

```
git clone https://github.com/Francis0121/nornenjs.git
cd nornenjs
node-gyp configure build
```

or install it via [npm](https://www.npmjs.org/):

```
npm install nornenjs
```



## Use libraries

- [Node.js](http://nodejs.org/)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone)
- [node-cuda](https://github.com/kashif/node-cuda)
- [BianryJS](http://binaryjs.com/)
- [socket.io](http://socket.io/)
- [Node-gyp](https://github.com/TooTallNate/node-gyp)
- [Jpeg](https://www.npmjs.com/package/jpeg)
- [Png](https://www.npmjs.com/package/png)
- [Sqlite3](https://github.com/mapbox/node-sqlite3)

