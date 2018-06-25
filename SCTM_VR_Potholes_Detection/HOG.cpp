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

typedef struct PointFloat{
    float x;
    float y;
} PointFloat;

typedef struct LineCoordinates{
    PointFloat startPoint;
    PointFloat endPoint;
} LineCoordinates;

LineCoordinates calculateLineCoordinates(const Point cellCenter,
                                         const PointFloat lineVecDirection,
                                         const float currentGradStrength,
                                         const float maxVectorialLength,
                                         const float scale){
        float startPointX = cellCenter.x - lineVecDirection.x * currentGradStrength * maxVectorialLength * scale;
        float startPointY = cellCenter.y - lineVecDirection.y * currentGradStrength * maxVectorialLength * scale;
        float endPointX = cellCenter.x + lineVecDirection.x * currentGradStrength * maxVectorialLength * scale;
        float endPointY = cellCenter.y + lineVecDirection.y * currentGradStrength * maxVectorialLength * scale;

        PointFloat startPoint = PointFloat{startPointX,startPointY};
        PointFloat endPoint = PointFloat{endPointX, endPointY};

        LineCoordinates result;
        result.startPoint =  startPoint;
        result.endPoint = endPoint;
        return result;
};

GradientLine calculateGradientLine(Mat visual_image,
                                   int binIndex,
                                   int scaleFactor,
                                   float currentGradStrength,
                                   float radRangeForOneBin,
                                   Size cellSize,
                                   double viz_factor,
                                   Point cellCenter) {
    float currRad = binIndex * radRangeForOneBin + radRangeForOneBin / 2;

    float vectorialDirectionX = cos(currRad);
    float vectorialDirectionY = sin(currRad);
    PointFloat vectorialDirection = PointFloat{vectorialDirectionX, vectorialDirectionY};

    float maxVectorialLength = cellSize.width / 2;
    float scale = viz_factor; // just a visual_imagealization scale,
    // to see the lines better

    // compute line coordinates
    LineCoordinates coordinates = calculateLineCoordinates(cellCenter,
                                                           vectorialDirection,
                                                           currentGradStrength,
                                                           maxVectorialLength,
                                                           scale);

    // draw gradient visual_imagealization
    GradientLine line = {visual_image,
                         Point(coordinates.startPoint.x * scaleFactor, coordinates.startPoint.y * scaleFactor),
                         Point(coordinates.endPoint.x * scaleFactor, coordinates.endPoint.y * scaleFactor),
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
                                  Point cellCenter,
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
                                                      cellCenter);
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
                                         Point cellCenter,
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
                                                      cellCenter);
            greaterGradient = line;
        }
    } // for (all bins)

    vector<GradientLine> vec = vector<GradientLine>();
    vec.push_back(greaterGradient);
    drawGradientLines(vec);
}


void drawCell(int cellsColumns,
              int cellsRows,
              Size cellSize,
              Mat visual_image,
              int scaleFactor,
              float radRangeForOneBin,
              double viz_factor,
              float ***gradientStrengths,
              int binNumber) {
    for (int cellRowIndex = 0; cellRowIndex < cellsRows; cellRowIndex++) {
        for (int cellColIndex = 0; cellColIndex < cellsColumns; cellColIndex++) {
            int cellBeginX = cellColIndex * cellSize.width;
            int cellBeginY = cellRowIndex * cellSize.height;

            int cellCenterX = cellBeginX + cellSize.width / 2;
            int cellCenterY = cellBeginY + cellSize.height / 2;
            Point cellCenter = Point(cellCenterX, cellCenterY);
//            rectangle(visual_image,
//                      Point(cellBeginX * scaleFactor, cellBeginY * scaleFactor),
//                      Point((cellBeginX + cellSize.width) * scaleFactor,
//                            (cellBeginY + cellSize.height) * scaleFactor),
//                      CV_RGB(100, 100, 100),
//                      1);

            drawGreaterGradientStrenghtsInCells(visual_image,
                                                scaleFactor,
                                                cellSize,
                                                radRangeForOneBin,
                                                gradientStrengths,
                                                cellRowIndex,
                                                cellColIndex,
                                                viz_factor,
                                                cellCenter,
                                                binNumber);

        }
    }
}

int computeGradientStrengthPerCell(int cellsColumns,
                                   int cellsRows,
                                   const vector<float> &descriptorValues,
                                   float ***gradientStrengths,
                                   int gradientBinSize,
                                   int **cellUpdateCounter) {
    // number of blocks = number of cells - 1
    // since there is a new block on each cellSize (overlapping blocks!) but the last one
    int blocksColumns = cellsColumns - 1;
    int blocksRows = cellsRows - 1;
    int descriptorDataIdx = 0;

    for (int blockColIndex = 0; blockColIndex < blocksColumns; blockColIndex++) {
        for (int blockRowIndex = 0; blockRowIndex < blocksRows; blockRowIndex++) {
            // 4 cells per block ...
            for (int cellNumber = 0; cellNumber < 4; cellNumber++) {
                // compute corresponding cellSize number
                int cellColIndex = blockColIndex;
                int cellRowIndex = blockRowIndex;
                if (cellNumber == 1) cellRowIndex++;
                if (cellNumber == 2) cellColIndex++;
                if (cellNumber == 3) {
                    cellColIndex++;
                    cellRowIndex++;
                }

                for (int bin = 0; bin < gradientBinSize; bin++) {
                    float gradientStrength = descriptorValues[descriptorDataIdx];
                    descriptorDataIdx++;

                    gradientStrengths[cellRowIndex][cellColIndex][bin] += gradientStrength;
                }

                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cellSize was updated,
                // to compute average gradient strengths
                cellUpdateCounter[cellRowIndex][cellColIndex]++;

            }


        } // for (all block x pos)
    } // for (all block y pos)

    return descriptorDataIdx;
}


void computeAverageGradientStrengths(int cellsColumns,
                                     int cellsRows,
                                     float ***gradientStrengths,
                                     int **cellUpdateCounter,
                                     int gradientBinSize) {
    for (int cellRowIndex = 0; cellRowIndex < cellsRows; cellRowIndex++) {
        for (int cellColIndex = 0; cellColIndex < cellsColumns; cellColIndex++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[cellRowIndex][cellColIndex];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < gradientBinSize; bin++) {
                gradientStrengths[cellRowIndex][cellColIndex][bin] /= NrUpdatesForThisCell;
            }
        }
    }
}

// HOGDescriptor visual_image analyzing
// Adapted from http://www.juergenbrauer.org/old_wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
// ONLY PRECAUSIONS ARE
// --> Image size width/heigth needs to be a multiple of block width/heigth
// --> Block size width/heigth (multiple cells) needs to be a multiple of cellSize size (histogram region) width/heigth
// --> Block stride needs to be a multiple of a cellSize size, however current code only allows to use a block stride = cellSize size!
// --> ScaleFactor enlarges the image patch to make it visible (e.g. a patch of 50x50 could have a factor 10 to be visible at scale 500x500 for inspection)
// --> viz_factor enlarges the maximum size of the maximal gradient length for normalization. At viz_factor = 1 it results in a length = half the cellSize width
Mat getHoGDescriptorVisualImage(const Mat &origImg,
                                const vector<float> &descriptorValues,
                                const Size cellSize,
                                const int scaleFactor,
                                const double viz_factor) {

    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

    int binNumber = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14 / (float) binNumber;

    Size winSize = origImg.size();
    int cellsColumns = winSize.width / cellSize.width;
    int cellsRows = winSize.height / cellSize.height;

    // prepare data structure: 9 orientation / gradient strenghts for each cellSize
    float ***gradientStrengths = new float **[cellsRows];
    int **cellUpdateCounter = new int *[cellsRows];
    for (int y = 0; y < cellsRows; y++) {
        gradientStrengths[y] = new float *[cellsColumns];
        cellUpdateCounter[y] = new int[cellsColumns];
        for (int x = 0; x < cellsColumns; x++) {
            gradientStrengths[y][x] = new float[binNumber];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < binNumber; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // compute gradient strengths per cell
    int descriptorDataIdx = computeGradientStrengthPerCell(cellsColumns,
                                                           cellsRows,
                                                           descriptorValues,
                                                           gradientStrengths,
                                                           binNumber,
                                                           cellUpdateCounter);


    // compute average gradient strengths
    computeAverageGradientStrengths(cellsColumns,
                                    cellsRows,
                                    gradientStrengths,
                                    cellUpdateCounter,
                                    binNumber);


    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    // draw cells
    drawCell(cellsColumns,
             cellsRows,
             cellSize,
             visual_image,
             scaleFactor,
             radRangeForOneBin,
             viz_factor,
             gradientStrengths,
             binNumber);


    // don't forget to free memory allocated by helper data structures!
    for (int y = 0; y < cellsRows; y++) {
        for (int x = 0; x < cellsColumns; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    return visual_image;

}