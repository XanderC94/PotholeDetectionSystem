//
// Created by Xander_C on 15/06/2018.
//

#include <opencv2/objdetect.hpp>
#include "HOG.h"

Mat get_hogdescriptor_visual_image(const Mat &src, const vector<float> &descriptors,
                                   const Size window, const Size cell, const int bins,
                                   const int scaling_factor, const double grad_viz,
                                   const bool isGrayscale, bool onlyMaxGrad) {
    Mat visual_image;
    if (scaling_factor == 1) {
        resize(src, visual_image, Size(src.cols * scaling_factor, src.rows * scaling_factor));
    }

    if (isGrayscale) {
        cvtColor(visual_image, visual_image, CV_GRAY2BGR);
    }

    // dividing 180� into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14f / (float) bins;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = window.width / cell.width;
    int cells_in_y_dir = window.height / cell.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[bins];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < bins; bin++)
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
                cellx = blockx;
                celly = blocky;

                if (cellNr == 1) celly++;
                if (cellNr == 2) cellx++;
                if (cellNr == 3) {
                    cellx++;
                    celly++;
                }

                for (int bin = 0; bin < bins; bin++) {
                    float gradientStrength = descriptors[descriptorDataIdx];
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
            for (int bin = 0; bin < bins; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }


    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    // draw cells
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cell.width;
            int drawY = celly * cell.height;

            int mx = drawX + cell.width / 2;
            int my = drawY + cell.height / 2;

            rectangle(visual_image,
                      Point(drawX * scaling_factor, drawY * scaling_factor),
                      Point((drawX + cell.width) * scaling_factor,
                            (drawY + cell.height) * scaling_factor),
                      CV_RGB(100, 100, 100),
                      1);

            if (onlyMaxGrad) {
                double maxGrad = 0.0;
                int idx = 0;

                for (int bin = 0; bin < bins; bin++) {
                    if (gradientStrengths[celly][cellx][bin] > maxGrad) {
                        maxGrad = gradientStrengths[celly][cellx][bin];
                        idx = bin;
                    }
                }

                // no line to draw?
                if (maxGrad == 0)
                    continue;

                float currRad = idx * radRangeForOneBin + radRangeForOneBin / 2;

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cell.width / 2;
                float scale = grad_viz; // just a visual_imagealization scale,
                // to see the lines better

                // compute line coordinates
                float x1 = mx - dirVecX * maxGrad * maxVecLen * scale;
                float y1 = my - dirVecY * maxGrad * maxVecLen * scale;
                float x2 = mx + dirVecX * maxGrad * maxVecLen * scale;
                float y2 = my + dirVecY * maxGrad * maxVecLen * scale;

                // draw gradient visual_imagealization
                line(visual_image,
                     Point(x1 * scale, y1 * scale),
                     Point(x2 * scale, y2 * scale),
                     CV_RGB(0, 0, 255),
                     1);
            } else {
                // draw in each cell all 9 gradient strengths
                for (int bin = 0; bin < bins; bin++) {
                    float currentGradStrength = gradientStrengths[celly][cellx][bin];

                    // no line to draw?
                    if (currentGradStrength == 0)
                        continue;

                    float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

                    float dirVecX = cos(currRad);
                    float dirVecY = sin(currRad);
                    float maxVecLen = cell.width / 2;
                    float scale = grad_viz; // just a visual_imagealization scale,
                    // to see the lines better

                    // compute line coordinates
                    float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                    float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                    float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                    float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                    // draw gradient visual_imagealization
                    line(visual_image,
                         Point(x1 * scale, y1 * scale),
                         Point(x2 * scale, y2 * scale),
                         CV_RGB(0, 0, 255),
                         1);

                } // for (all bins)
            }

        } // for (cellx)
    } // for (celly)


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

void HOG(const Mat &src, vector<float> &descriptors, vector<Point> &locations, const HOGConfig config) {

    Mat grayscale;
    cv::HOGDescriptor hog = HOGDescriptor(config.window, config.block, config.block_stride, config.cell, config.bin);

    cvtColor(src, grayscale, CV_BGR2GRAY);

    cout << "Computing HOG Descriptor...";
    hog.compute(grayscale, descriptors, config.window_stride, config.padding, locations);
}