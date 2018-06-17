//
// Created by Xander_C on 15/06/2018.
//

#include <opencv2/objdetect.hpp>
#include "HOG.h"


void freeMemory(int cells_in_x_dir, int cells_in_y_dir, float ***gradientStrengths, int **cellUpdateCounter) {
    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
}

void drawGradientStrengthsInBin(Mat &visual_image,
                                float ***gradientStrengths,
                                int cellx,
                                int celly,
                                int bin,
                                float radRangeForOneBin,
                                Size cellSize,
                                int scaleFactor,
                                double viz_factor,
                                int mx,
                                int my) {

    float currentGradStrength = gradientStrengths[celly][cellx][bin];

    // no line to draw?
    if (currentGradStrength != 0) {
        float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

        float dirVecX = cos(currRad);
        float dirVecY = sin(currRad);
        float maxVecLen = cellSize.width / 2;
        float scale = (float) viz_factor; // just a visual_imagealization scale,
        // to see the lines better

        // compute line coordinates
        float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
        float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
        float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
        float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

        // draw gradient visual_imagealization
        line(visual_image,
             Point(x1 * scaleFactor, y1 * scaleFactor),
             Point(x2 * scaleFactor, y2 * scaleFactor),
             CV_RGB(0, 0, 255),
             1);
    }
}

/*
 * function refactored base on:
 *  http://www.juergenbrauer.org/old_wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
 * */
Mat get_hogdescriptor_visual_image(Mat &origImg,
                                   vector<float> &descriptorValues,
                                   Size winSize,
                                   Size cellSize,
                                   int scaleFactor,
                                   double viz_factor) {
    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14 / (float) gradientBinSize;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
        for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
            // 4 cells per block ...
            for (int cellNr = 0; cellNr < 4; cellNr++) {
                // compute corresponding cell nr
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
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;

            } // for (all cells)


        } // for (all block x pos)
    } // for (all block y pos)


    // compute average gradient strengths
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < gradientBinSize; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }


    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    // draw cells
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;

            int mx = drawX + cellSize.width / 2;
            int my = drawY + cellSize.height / 2;

            rectangle(visual_image,
                      Point(drawX * scaleFactor, drawY * scaleFactor),
                      Point((drawX + cellSize.width) * scaleFactor,
                            (drawY + cellSize.height) * scaleFactor),
                      CV_RGB(100, 100, 100),
                      1);

            // draw in each cell all 9 gradient strengths
            for (int bin = 0; bin < gradientBinSize; bin++) {
                drawGradientStrengthsInBin(visual_image, gradientStrengths,
                                           cellx, celly, bin, radRangeForOneBin, cell, grad_viz, mx, my);
            } // for (all bins)

        } // for (cellx)
    } // for (celly)


    // don't forget to free memory allocated by helper data structures!
    freeMemory(cells_in_x_dir, cells_in_y_dir, gradientStrengths, cellUpdateCounter);

    return visual_image;

}


void HOG(const Mat &src, vector<float> &descriptors, vector<Point> &locations, const HOGConfig config) {

    Mat grayscale;
    cv::HOGDescriptor hog;// = HOGDescriptor(config.window, config.block, config.block_stride, config.cell, config.bin);

    cvtColor(src, grayscale, CV_BGR2GRAY);

    cout << "Computing HOG Descriptor...";
    hog.compute(grayscale, descriptors, config.window_stride, config.padding, locations);
}