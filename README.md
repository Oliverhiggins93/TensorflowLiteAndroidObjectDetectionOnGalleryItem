# TensorflowLiteAndroidObjectDetectionOnGalleryItem
This is an app for Android that opens an image from gallery, crops it to 300 x 300 pixels and then runs tensorflow image detection on the image.

The reason for the crop is so that it can be run on information-dense microscopy images. 
The included model is named Onioncam - it's simply trained on images of Onion skins taken using a 1x microscope. 
The model was trained using MobileNet-V2 SSD (unquantized) and compressed to a TFLite file. 
