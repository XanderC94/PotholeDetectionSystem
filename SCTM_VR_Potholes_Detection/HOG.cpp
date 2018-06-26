//
// Created by Xander_C on 15/06/2018.
//

#include "HOG.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

HoG calculateHoG(const Mat &src, const HOGConfig config) {

    Mat grayscale;
    cv::HOGDescriptor hog(config.detenctionWindowSize,
                          config.blockSize,
                          config.blockStride,
                          config.cellSize,
                          config.binNumber);

    cvtColor(src, grayscale, CV_BGR2GRAY);
    cout << "Computing HOG Descriptor..." << endl;

    vector<float> descriptors = vector<float>();
    vector<Point> locations = vector<Point>();
    hog.compute(grayscale, descriptors, Size(0, 0), Size(0, 0), locations);

    HoG result = {descriptors, locations};
    return result;
}

typedef struct GradientLine {
    Mat visual_image;
    Point startPoint;
    Point endPoint;
    Scalar color;
    int thickness;
} GradienLine;

GradientLine calculateGradientLine(Mat visual_image,
                                   int binIndex,
                                   int scaleFactor,
                                   float currentGradStrength,
                                   float radRangeForOneBin,
                                   Size cellSize,
                                   double viz_factor,
                                   int mx,
                                   int my) {
    float currRad = binIndex * radRangeForOneBin + radRangeForOneBin / 2;

    float dirVecX = cos(currRad);
    float dirVecY = sin(currRad);
    float maxVecLen = cellSize.width / 2;
    float scale = viz_factor; // just a visual_imagealization scale,
    // to see the lines better

    // compute line coordinates
    float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
    float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
    float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
    float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

    // draw gradient visual_imagealization
    GradientLine line = {visual_image,
                         Point(x1 * scaleFactor, y1 * scaleFactor),
                         Point(x2 * scaleFactor, y2 * scaleFactor),
                         CV_RGB(0, 0, 255),
                         1};
    return line;
}

void drawGradientLines(vector<GradientLine> lines) {
    for (GradientLine gLine : lines) {
        line(gLine.visual_image,
             gLine.startPoint,
             gLine.endPoint,
             gLine.color,
             gLine.thickness);
    }
}

void drawGradientStrenghtsInCells(Mat visual_image,
                                  int scaleFactor,
                                  Size cellSize,
                                  float radRangeForOneBin,
                                  float ***gradientStrengths,
                                  int celly,
                                  int cellx,
                                  double viz_factor,
                                  int mx,
                                  int my,
                                  int binNumber) {

    vector<GradienLine> gLines = vector<GradienLine>();

    for (int bin = 0; bin < binNumber; bin++) {
        float currentGradStrength = gradientStrengths[celly][cellx][bin];

        // no line to draw?
        if (currentGradStrength != 0) {
            GradientLine line = calculateGradientLine(visual_image,
                                                      bin,
                                                      scaleFactor,
                                                      currentGradStrength,
                                                      radRangeForOneBin,
                                                      cellSize,
                                                      viz_factor,
                                                      mx,
                                                      my);
            gLines.push_back(line);
        }
    } // for (all bins)

    drawGradientLines(gLines);
}

void drawGreaterGradientStrenghtsInCells(Mat visual_image,
                                         int scaleFactor,
                                         Size cellSize,
                                         float radRangeForOneBin,
                                         float ***gradientStrengths,
                                         int celly,
                                         int cellx,
                                         double viz_factor,
                                         int mx,
                                         int my,
                                         int binNumber) {

    GradienLine greaterGradient = {Mat(), Point(), Point(), Scalar(), 0};

    float prevGradStrength = 0.0;

    for (int bin = 0; bin < binNumber; bin++) {
        float currentGradStrength = gradientStrengths[celly][cellx][bin];

        // no line to draw?
        if (currentGradStrength != 0 && prevGradStrength < currentGradStrength) {
            prevGradStrength = currentGradStrength;
            GradientLine line = calculateGradientLine(visual_image,
                                                      bin,
                                                      scaleFactor,
                                                      currentGradStrength,
                                                      radRangeForOneBin,
                                                      cellSize,
                                                      viz_factor,
                                                      mx,
                                                      my);
            greaterGradient = line;
        }
    } // for (all bins)

    vector<GradientLine> vec = vector<GradientLine>();
    vec.push_back(greaterGradient);
    drawGradientLines(vec);
}


void drawCell(int cells_in_x_dir,
              int cells_in_y_dir,
              Size cellSize,
              Mat visual_image,
              int scaleFactor,
              float radRangeForOneBin,
              double viz_factor,
              float ***gradientStrengths,
              int binNumber) {
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;

            int mx = drawX + cellSize.width / 2;
            int my = drawY + cellSize.height / 2;

//            rectangle(visual_image,
//                      Point(drawX * scaleFactor, drawY * scaleFactor),
//                      Point((drawX + cellSize.width) * scaleFactor,
//                            (drawY + cellSize.height) * scaleFactor),
//                      CV_RGB(100, 100, 100),
//                      1);

            drawGreaterGradientStrenghtsInCells(visual_image,
                                                scaleFactor,
                                                cellSize,
                                                radRangeForOneBin,
                                                gradientStrengths,
                                                celly,
                                                cellx,
                                                viz_factor,
                                                mx,
                                                my,
                                                binNumber);

        } // for (cellx)
    } // for (celly)
}

int computeGradientStrengthPerCell(int cells_in_x_dir,
                                   int cells_in_y_dir,
                                   vector<float> &descriptorValues,
                                   float ***gradientStrengths,
                                   int gradientBinSize,
                                   int **cellUpdateCounter) {
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cellSize (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
        for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
            // 4 cells per block ...
            for (int cellNr = 0; cellNr < 4; cellNr++) {
                // compute corresponding cellSize nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr == 1) celly++;
                if (cellNr == 2) cellx++;
                if (cellNr == 3) {
                    cellx++;
                    celly++;
                }

                for (int bin = 0; bin < gradientBinSize; bin++) {
                    float gradientStrength = descriptorValues[descriptorDataIdx];
                    descriptorDataIdx++;

                    gradientStrengths[celly][cellx][bin] += gradientStrength;

                } // for (all bins)


                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cellSize was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;

            } // for (all cells)


        } // for (all block x pos)
    } // for (all block y pos)

    return descriptorDataIdx;
}


void computeAverageGradientStrengths(int cells_in_x_dir,
                                     int cells_in_y_dir,
                                     float ***gradientStrengths,
                                     int **cellUpdateCounter,
                                     int gradientBinSize) {
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < gradientBinSize; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
}

// HOGDescriptor visual_image analyzing
// Adapted from http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
// ONLY PRECAUSIONS ARE
// --> Image size width/heigth needs to be a multiple of block width/heigth
// --> Block size width/heigth (multiple cells) needs to be a multiple of cellSize size (histogram region) width/heigth
// --> Block stride needs to be a multiple of a cellSize size, however current code only allows to use a block stride = cellSize size!
// --> ScaleFactor enlarges the image patch to make it visible (e.g. a patch of 50x50 could have a factor 10 to be visible at scale 500x500 for inspection)
// --> viz_factor enlarges the maximum size of the maximal gradient length for normalization. At viz_factor = 1 it results in a length = half the cellSize width
Mat getHoGDescriptorVisualImage(Mat &origImg,
                                vector<float> &descriptorValues,
                                Size winSize,
                                Size cellSize,
                                int scaleFactor,
                                double viz_factor) {
    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

    int binNumber = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14 / (float) binNumber;

    // prepare data structure: 9 orientation / gradient strenghts for each cellSize
    int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[binNumber];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < binNumber; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // compute gradient strengths per cell
    int descriptorDataIdx = computeGradientStrengthPerCell(cells_in_x_dir,
                                                           cells_in_y_dir,
                                                           descriptorValues,
                                                           gradientStrengths,
                                                           binNumber,
                                                           cellUpdateCounter);


    // compute average gradient strengths
    computeAverageGradientStrengths(cells_in_x_dir, cells_in_y_dir, gradientStrengths, cellUpdateCounter,
                                    binNumber);


    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    // draw cells
    drawCell(cells_in_x_dir,
             cells_in_y_dir,
             cellSize,
             visual_image,
             scaleFactor,
             radRangeForOneBin,
             viz_factor,
             gradientStrengths,
             binNumber);


    // don't forget to free memory allocated by helper data structures!
    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    return visual_image;

}