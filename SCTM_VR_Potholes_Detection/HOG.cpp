//
// Created by Xander_C on 15/06/2018.
//

#include "HOG.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

HOG calculateHOG(const Mat &src, const HOGConfig config) {

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

    HOG result = {descriptors, locations};
    return result;
}

OrientedGradient createOrientedGradient(float gradStrength, float radRangeForOneBin, int binIndex) {
    float directionInRadians = binIndex * radRangeForOneBin + radRangeForOneBin / 2;
    OrientedGradient result{gradStrength, directionInRadians};
    return result;
}

PointFloat convertRadiansToCartesian(float radiant) {
    float vectorialDirectionX = cos(radiant);
    float vectorialDirectionY = sin(radiant);
    PointFloat result = PointFloat{vectorialDirectionX, vectorialDirectionY};
    return result;
}

LineCoordinates calculateLineCoordinates(const Point cellCenter,
                                         OrientedGradient oGradient,
                                         const float maxVectorialLength,
                                         const float scale){
    PointFloat vectorialDirection = convertRadiansToCartesian(oGradient.directionInRadians);

    float startPointX = cellCenter.x - vectorialDirection.x * oGradient.strength * maxVectorialLength * scale;
    float startPointY = cellCenter.y - vectorialDirection.y * oGradient.strength * maxVectorialLength * scale;
    float endPointX = cellCenter.x + vectorialDirection.x * oGradient.strength * maxVectorialLength * scale;
    float endPointY = cellCenter.y + vectorialDirection.y * oGradient.strength * maxVectorialLength * scale;

        PointFloat startPoint = PointFloat{startPointX,startPointY};
        PointFloat endPoint = PointFloat{endPointX, endPointY};

        LineCoordinates result;
        result.startPoint =  startPoint;
        result.endPoint = endPoint;
        return result;
}


GradientLine calculateGradientLine(OrientedGradientInCell oGradientInCell,
                                   int scaleFactor,
                                   Size cellSize,
                                   double viz_factor) {

    float maxVectorialLength = cellSize.width / 2;
    float scale = viz_factor; // just a visual_imagealization scale,
    // to see the lines better

    // compute line coordinates
    LineCoordinates coordinates = calculateLineCoordinates(oGradientInCell.cellCenter,
                                                           oGradientInCell.orientedGradientValue,
                                                           maxVectorialLength,
                                                           scale);

    GradientLine line = {Point(coordinates.startPoint.x * scaleFactor, coordinates.startPoint.y * scaleFactor),
                         Point(coordinates.endPoint.x * scaleFactor, coordinates.endPoint.y * scaleFactor),
                         CV_RGB(0, 0, 255),
                         1};
    return line;
}

void drawGradientLines(Mat srcImage, vector<GradientLine> lines) {
    for (GradientLine gLine : lines) {
        line(srcImage,
             gLine.startPoint,
             gLine.endPoint,
             gLine.color,
             gLine.thickness);
    }
}

vector<OrientedGradient> getOrientedGradientsInCell(int cellx,
                                                    int celly,
                                                    float ***gradientStrengths,
                                                    int binNumber,
                                                    float radRangeForOneBin) {
    vector<OrientedGradient> result = vector<OrientedGradient>();
    for (int binIndex = 0; binIndex < binNumber; binIndex++) {
        float currentGradStrength = gradientStrengths[celly][cellx][binIndex];
        if (currentGradStrength != 0) {
            OrientedGradient oGradient = createOrientedGradient(currentGradStrength, radRangeForOneBin, binIndex);
            result.push_back(oGradient);
        }
    }
    return result;
}

OrientedGradient getGreaterOrientedGradientInCell(int cellx,
                                                  int celly,
                                                  float ***gradientStrengths,
                                                  int binNumber,
                                                  float radRangeForOneBin) {
    OrientedGradient greaterGradient{0.0, 0.0};
    float prevGradStrength = 0.0;
    for (int binIndex = 0; binIndex < binNumber; binIndex++) {
        float currentGradStrength = gradientStrengths[celly][cellx][binIndex];
        // no line to draw?
        if (prevGradStrength < currentGradStrength) {
            prevGradStrength = currentGradStrength;
            greaterGradient = createOrientedGradient(currentGradStrength, radRangeForOneBin, binIndex);
        }
    }
    return greaterGradient;
}


void drawGradientStrengthInCells(Mat visual_image,
                                 OrientedGradientInCell gradientInCells,
                                 int scaleFactor,
                                 Size cellSize,
                                 double viz_factor) {

    GradientLine greaterGradientLine = calculateGradientLine(gradientInCells,
                                                             scaleFactor,
                                                             cellSize,
                                                             viz_factor);

    vector<GradientLine> vec = vector<GradientLine>();
    vec.push_back(greaterGradientLine);
    drawGradientLines(visual_image, vec);
}

vector<OrientedGradientInCell> computeOrientedGradientsCellsVector(int cellsRows,
                                                                   int cellsColumns,
                                                                   Size cellSize,
                                                                   float radRangeForOneBin,
                                                                   float ***gradientStrengths,
                                                                   int binNumber) {
    vector<OrientedGradientInCell> result = vector<OrientedGradientInCell>();
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

            vector<OrientedGradient> orietedGradients = getOrientedGradientsInCell(cellColIndex,
                                                                                   cellRowIndex,
                                                                                   gradientStrengths,
                                                                                   binNumber,
                                                                                   radRangeForOneBin);
            for (OrientedGradient og : orietedGradients) {
                OrientedGradientInCell graterGradientInCell{og, cellCenter};
                result.push_back(graterGradientInCell);
            }
        }
    }

    return result;
}

vector<OrientedGradientInCell> computeGreaterOrientedGradientCellsVector(int cellsRows,
                                                                         int cellsColumns,
                                                                         Size cellSize,
                                                                         float radRangeForOneBin,
                                                                         float ***gradientStrengths,
                                                                         int binNumber) {
    vector<OrientedGradientInCell> result = vector<OrientedGradientInCell>();
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

            OrientedGradient greaterGradient = getGreaterOrientedGradientInCell(cellColIndex,
                                                                                cellRowIndex,
                                                                                gradientStrengths,
                                                                                binNumber,
                                                                                radRangeForOneBin);
            OrientedGradientInCell graterGradientInCell{greaterGradient, cellCenter};

            result.push_back(graterGradientInCell);
        }
    }

    return result;
}

void drawCell(vector<OrientedGradientInCell> greaterOrientedGradientCells,
              Size cellSize,
              Mat visual_image,
              int scaleFactor,
              double viz_factor) {
    for (OrientedGradientInCell ogInCell : greaterOrientedGradientCells) {
        drawGradientStrengthInCells(visual_image, ogInCell, scaleFactor, cellSize, viz_factor);
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


void initializeMemory(float ***gradientStrengths,
                      int **cellUpdateCounter,
                      const int cellsRows,
                      const int cellsColumns,
                      int binNumber) {
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
}

void cleanMemory(float ***gradientStrengths,
                 int **cellUpdateCounter,
                 const int cellsRows,
                 const int cellsColumns) {
    for (int y = 0; y < cellsRows; y++) {
        for (int x = 0; x < cellsColumns; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
}

vector<OrientedGradientInCell> computeHOGCells(const Mat origImg,
                                               const vector<float> &descriptorValues,
                                               const Size cellSize) {
    Size winSize = origImg.size();
    // prepare data structure: 9 orientation / gradient strenghts for each cellSize
    int binNumber = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14 / (float) binNumber;

    int cellsColumns = winSize.width / cellSize.width;
    int cellsRows = winSize.height / cellSize.height;

    float ***gradientStrengths = new float **[cellsRows];
    int **cellUpdateCounter = new int *[cellsRows];
    initializeMemory(gradientStrengths, cellUpdateCounter, cellsRows, cellsColumns, binNumber);

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


    //cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    vector<OrientedGradientInCell> greaterOrientedGradientCells = computeOrientedGradientsCellsVector(cellsRows,
                                                                                                      cellsColumns,
                                                                                                      cellSize,
                                                                                                      radRangeForOneBin,
                                                                                                      gradientStrengths,
                                                                                                      binNumber);
    // don't forget to free memory allocated by helper data structures!
    cleanMemory(gradientStrengths, cellUpdateCounter, cellsRows, cellsColumns);

    return greaterOrientedGradientCells;
}

vector<OrientedGradientInCell> computeGreaterHOGCells(const Mat origImg,
                                                      const vector<float> &descriptorValues,
                                                      const Size cellSize) {
    Size winSize = origImg.size();
    // prepare data structure: 9 orientation / gradient strenghts for each cellSize
    int binNumber = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14 / (float) binNumber;

    int cellsColumns = winSize.width / cellSize.width;
    int cellsRows = winSize.height / cellSize.height;

    float ***gradientStrengths = new float **[cellsRows];
    int **cellUpdateCounter = new int *[cellsRows];
    initializeMemory(gradientStrengths, cellUpdateCounter, cellsRows, cellsColumns, binNumber);

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


    //cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    vector<OrientedGradientInCell> greaterOrientedGradientCells = computeGreaterOrientedGradientCellsVector(cellsRows,
                                                                                                            cellsColumns,
                                                                                                            cellSize,
                                                                                                            radRangeForOneBin,
                                                                                                            gradientStrengths,
                                                                                                            binNumber);
    // don't forget to free memory allocated by helper data structures!
    cleanMemory(gradientStrengths, cellUpdateCounter, cellsRows, cellsColumns);

    return greaterOrientedGradientCells;
}

// HOGDescriptor visual_image analyzing
// Adapted from http://www.juergenbrauer.org/old_wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
// ONLY PRECAUSIONS ARE
// --> Image size width/heigth needs to be a multiple of block width/heigth
// --> Block size width/heigth (multiple cells) needs to be a multiple of cellSize size (histogram region) width/heigth
// --> Block stride needs to be a multiple of a cellSize size, however current code only allows to use a block stride = cellSize size!
// --> ScaleFactor enlarges the image patch to make it visible (e.g. a patch of 50x50 could have a factor 10 to be visible at scale 500x500 for inspection)
// --> viz_factor enlarges the maximum size of the maximal gradient length for normalization. At viz_factor = 1 it results in a length = half the cellSize width
Mat overlapOrientedGradientCellsOnImage(const Mat &origImg,
                                        vector<OrientedGradientInCell> greaterOrientedGradientCells,
                                        const Size cellSize,
                                        const int scaleFactor,
                                        const double viz_factor) {

    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));
    // draw cells
    drawCell(greaterOrientedGradientCells, cellSize, visual_image, scaleFactor, viz_factor);
    return visual_image;
}

vector<OrientedGradientInCell> selectNeighbourhoodCellsAtContour(Mat contoursMask,
                                                                 vector<OrientedGradientInCell> orientedGradientsCells,
                                                                 int neighbourhood){
    vector<cv::Point> contourPoints;
    findNonZero(contoursMask, contourPoints);
    vector<OrientedGradientInCell> result;
    for(OrientedGradientInCell oGCell : orientedGradientsCells){
        bool alreadyAdded = false;
        for(Point sPPoint : contourPoints){
            double distance = cv::norm(oGCell.cellCenter - sPPoint);
            if(distance <= neighbourhood && !alreadyAdded){
                result.push_back(oGCell);
                alreadyAdded = true;
            }
        }
    }
    return result;
}