package com.dji.sdk.sample.demo.flightcontroller;

import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.*;

import java.util.ArrayList;
import java.util.List;


public class Targeting {
    String modelWeights = "D:\\OpenCV\\yolov3-tiny.weights";
    String modelConfiguration = "D:\\OpenCV\\yolov3-tiny.cfg";
    final Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
    private final int INPUT_WIDTH = 288;
    private final int INPUT_HEIGHT = 288;
    private final double IMG_SCALING = 0.00392;

    private static final int EXCLUSION_LEFT_BOUND = 400;
    private static final int EXCLUSION_RIGHT_BOUND = 1520;

    public Targeting() {
    }

    public void analyzeVideo(byte[] videoBuffer) {
        runImage(videoBuffer, net);
    }

    public void runImage(byte[]videoBuffer,Net net) {
        //TODO get DJI Camera data here

        Mat curFrame = byteVidToMat(videoBuffer);
        analyzeFrame(curFrame, net);
    }

    private List<String> getOutputNames(Net net) { //not mine
        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        for(Integer i : outLayers){
            names.add(layersNames.get(i-1));
        }
        //outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }


    private void analyzeFrame(Mat frame, Net net) {
        Size sz = new Size(INPUT_WIDTH, INPUT_HEIGHT);
        List<Mat> result = new ArrayList<>();
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect2d> rects = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);

        int centerX;
        int centerY;
        int width;
        int height;
        int left;
        int top;
        Mat row;
        Mat scores;
        Mat level;
        Core.MinMaxLocResult mm;
        float confidence;
        Point classIdPoint;

        Mat blob = Dnn.blobFromImage(frame, IMG_SCALING, sz, new Scalar(0), true);  //edit this maybe, scalar is empty rn, so no mean subtraction i think
        net.setInput(blob);
        net.forward(result, outBlobNames); //result is a 4d tensor, images, height, width, color channels

        float confThreshold = 0.6f; // Insert thresholding beyond which the model will detect objects//

        for (int i = 0; i < result.size(); ++i) {
            // each row is a candidate detection, the 1st 4 numbers are
            // [center_x, center_y, width, height], followed by (N-4) class probabilities

            level = result.get(i); //gets i output blob image from network, now a 3d tensor, height, width, color channels

            for (int j = 0; j < level.rows(); ++j) {
                row = level.row(j); // gets the data for the image,
                scores = row.colRange(5, level.cols()); //scores are class probabilities listed after the first 4 values
                mm = Core.minMaxLoc(scores); //finds maximum score in class probabilities
                confidence = (float) mm.maxVal; //confidence = maxVal
                classIdPoint = mm.maxLoc; //Id index is the index of maxVal

                if (confidence > confThreshold) {
                    centerX = (int) (row.get(0, 0)[0] * frame.cols()); //gets centerX from output blob and scales it for the input image
                    centerY = (int) (row.get(0, 1)[0] * frame.rows()); //same but with centerY
                    width = (int) (row.get(0, 2)[0] * frame.cols()); //same but with width
                    height = (int) (row.get(0, 3)[0] * frame.rows() + 100);//same but with height
                    left = centerX - width / 2;
                    top = centerY - height / 2;

                    clsIds.add((int) classIdPoint.x);
                    confs.add((float) confidence);
                    rects.add(new Rect2d(left, top, width, height));
                }
            }
        }

        float nmsThresh = 0.5f;
        MatOfFloat confidences;
        if (!confs.isEmpty()) {
            confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));

            Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
            MatOfRect2d boxes = new MatOfRect2d();
            boxes.fromArray(boxesArray);
            MatOfInt indices = new MatOfInt();
            Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); //eliminates weaker classifications of same objects

            //System.out.println(clsIds.get(0));
            int[] ind = indices.toArray();
            for (int idx : ind) {
                Rect2d box = boxesArray[idx];
                Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 255, 0), 2);
            }
            //drawLargestBox(boxesArray, ind, frame); //only biggest object drawn
        }
    }

    private void drawLargestBox(Rect2d[] boxes, int[] indices, Mat frame) {
        double largestArea = 0;
        Rect2d largestBox = null;
        Rect2d curBox;
        for (int index : indices) {
            curBox = boxes[index];
            if (curBox.x >= EXCLUSION_LEFT_BOUND && curBox.x <= EXCLUSION_RIGHT_BOUND) {
                if (curBox.area() >= largestArea) {
                    largestBox = curBox;
                    largestArea = curBox.area();
                }
            }
        }
        assert largestBox != null;
        Imgproc.rectangle(frame, largestBox.tl(), largestBox.br(), new Scalar(255, 200, 0), 2); //scalar is color (B, G, R)
    }


    private Mat byteVidToMat(byte[] vidFrame){
        if(vidFrame != null) {
            return new MatOfByte(vidFrame);
        }
        return null;
    }
}



