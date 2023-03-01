/*
 *  Copyright (C) 2023 Texas Instruments Incorporated - http://www.ti.com/
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Third-party headers. */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/* Module headers. */
#include <common/include/post_process_image_object_6d_pose_estimation.h>

namespace ti::edgeai::common
{
using namespace cv;
using namespace std;

vector<vector<int>> COLOR_MAP       = {{0,113,188},{216,82,24},{236,176,31},
                                       {125,46,141},{118,171,47},{76,189,237},
                                       {161,19,46},{76,76,76},{153,153,153},
                                       {255,0,0},{255,127,0},{190,190,0},
                                       {0,255,0},{0,0,255},{170,0,255},
                                       {84,84,0},{84,170,0},{84,255,0},
                                       {170,84,0},{170,170,0},{170,255,0},
                                       {255,84,0},{255,170,0},{255,255,0},
                                       {0,84,127},{0,170,127},{0,255,127},
                                       {84,0,127},{84,84,127},{84,170,127},
                                       {84,255,127},{170,0,127},{170,84,127},
                                       {170,170,127},{170,255,127},{255,0,127},
                                       {255,84,127},{255,170,127},{255,255,127},
                                       {0,84,255},{0,170,255},{0,255,255},
                                       {84,0,255},{84,84,255},{84,170,255},
                                       {84,255,255},{170,0,255},{170,84,255},
                                       {170,170,255},{170,255,255},{255,0,255},
                                       {255,84,255},{255,170,255},{84,0,0},
                                       {127,0,0},{170,0,0},{212,0,0},
                                       {255,0,0},{0,42,0},{0,84,0},
                                       {0,127,0},{0,170,0},{0,212,0},
                                       {0,255,0},{0,0,42},{0,0,84},
                                       {0,0,127},{0,0,170},{0,0,212},
                                       {0,0,255},{0,0,0},{36,36,36},
                                       {72,72,72},{109,109,109},{145,145,145},
                                       {182,182,182},{218,218,218},{0,113,188},
                                       {80,182,188},{127,127,0}};

vector<vector<float>> YCBV_CAMERA_MATRIX = {{1066.778, 0, 312.9869},
                                           {0.0, 1067.487, 241.3109},
                                           {0.0, 0.0, 1.0}};

vector<vector<float>> LM_CAMERA_MATRIX  = {{572.4114, 0.0, 325.2611},
                                           {0.0, 573.57043, 242.04899},
                                           {0.0, 0.0, 1.0}};

vector<vector<float>> YCBV_VERTICES      = {{51.1445, 51.223, 70.072},
                                           {35.865, 81.9885, 106.743},
                                           {24.772, 47.024, 88.0075},
                                           {33.927, 33.875, 51.0185},
                                           {48.575, 33.31, 95.704},
                                           {42.755, 42.807, 16.7555},
                                           {68.924, 64.3955, 19.414},
                                           {44.6775, 50.5545, 15.06},
                                           {51.0615, 30.161, 41.8185},
                                           {54.444, 89.206, 18.335},
                                           {74.4985, 72.3845, 121.32},
                                           {51.203, 33.856, 125.32},
                                           {80.722, 80.5565, 27.485},
                                           {58.483, 46.5375, 40.692},
                                           {92.1205, 93.717, 28.6585},
                                           {51.9755, 51.774, 102.945},
                                           {48.04, 100.772, 7.858},
                                           {10.5195, 60.4225, 9.4385},
                                           {59.978, 85.639, 19.575},
                                           {104.897, 82.18, 18.1665},
                                           {26.315, 38.921, 25.5655}};

vector<vector<float>> LM_VERTICES      =  {{-37.93430000, 38.79960000, 45.88450000},
                                           {107.83500000, 60.92790000, 109.70500000},
                                           {83.21620000, 82.65910000, 37.23640000},
                                           {68.32970000, 71.51510000, 50.24850000},
                                           {50.39580000,  90.89790000,  96.86700000},
                                           {33.50540000,  63.81650000,  58.72830000},
                                           {58.78990000,  45.75560000,  47.31120000},
                                           {114.73800000,  37.73570000,  104.00100000},
                                           {52.21460000,  38.70380000,  42.84850000},
                                           {75.09230000,  53.53750000,  34.62070000},
                                           {18.36050000,  38.93300000,  86.40790000},
                                           {50.44390000,  54.24850000,  45.40000000},
                                           {129.11300000,  59.24100000,  70.56620000},
                                           {101.57300000,  58.87630000,  106.55800000},
                                           {46.95910000,  73.71670000,  92.37370000}};

vector<vector<float>> vertices_order    = {{-1, -1, -1},
                                           {-1, -1, 1},
                                           {-1, 1,  1},
                                           {-1, 1, -1},
                                           {1, -1, -1},
                                           {1, -1,  1},
                                           {1,  1,  1},
                                           {1,  1, -1}};

PostprocessImageObject6DPoseEstimation::PostprocessImageObject6DPoseEstimation(const PostprocessImageConfig  &config,
                                                                               const DebugDumpConfig         &debugConfig):
    PostprocessImage(config,debugConfig)
{
    m_scaleX = static_cast<float>(m_config.outDataWidth)/m_config.inDataWidth;
    m_scaleY = static_cast<float>(m_config.outDataHeight)/m_config.inDataHeight;
    vector<vector<float>> *vertex;
    if (m_config.dataset == "ycbv")
    {
        m_cameraMatrix = &YCBV_CAMERA_MATRIX;
        vertex = &YCBV_VERTICES;
    }
    else
    {
        m_cameraMatrix = &LM_CAMERA_MATRIX;
        vertex = &LM_VERTICES;
    }

    int vertex_size = (int)vertex->capacity();
    for (int i = 0 ; i < vertex_size ; i++)
    {
        vector<vector<float>> temp_vector_1 = {};
        for (unsigned j = 0; j < vertices_order.size() ; j++)
        {
            vector<float> temp_vector_2 = {};
            for (unsigned k = 0; k < vertices_order[j].size() ; k++)
            {
                temp_vector_2.push_back(vertices_order[j][k]*vertex->at(i)[k]);
            }
            temp_vector_1.push_back(temp_vector_2);
        }
        m_vertices.push_back(temp_vector_1);
    }

}

static void matrix_multiply(vector<vector<float>> &mat1,
                            vector<vector<float>> &mat2,
                            vector<vector<float>> &result)
{ 
    int R1 = mat1.size();
    int C1 = mat1[0].size();
    int R2 = mat2.size();
    int C2 = mat2[0].size();
    if (C1 != R2)
    {
        return;
    }

    for (int i = 0; i < R1; i++) {
        vector<float> column{};
        for (int j = 0; j < C2; j++) {
            float value = 0;
            for (int k = 0; k < R2; k++) {
                value += mat1[i][k] * mat2[k][j];
            }
            column.push_back(value);
        }
        result.push_back(column);
     }
}

static void cross_product_and_transpose(vector<float>         &mat1,
                                        vector<float>         &mat2,
                                        vector<vector<float>> &result)
{
    float val1 = (mat1[1] * mat2[2]) - (mat1[2] * mat2[1]);
    float val2 = -((mat1[0] * mat2[2]) - (mat1[2] * mat2[0]));
    float val3 = (mat1[0] * mat2[1]) - (mat1[1] * mat2[0]);

    result.push_back({mat1[0],mat1[1],mat1[2]});
    result.push_back({mat2[0],mat2[1],mat2[2]});
    result.push_back({val1,val2,val3});
}

static void get_cuboid_corners_2d(vector<vector<float>> &vertices,
                                  vector<vector<float>> &rotation_matrix,
                                  vector<float>         &translation_vector,
                                  vector<vector<float>> *m_cameraMatrix,
                                  vector<vector<float>> &cuboid_corners)
{
    vector<vector<float>> result_matrix;
    matrix_multiply(vertices,rotation_matrix,result_matrix);

    for (unsigned i = 0; i<result_matrix.size(); i++)
    {
        for (unsigned j = 0; j < result_matrix[i].size(); j++)
        {
            result_matrix[i][j] +=  translation_vector[j];
        }

        for (unsigned j = 0; j < result_matrix[i].size(); j++)
        {
            result_matrix[i][j] /=  result_matrix[i][2];
        }
    }

    //Transpose Camera Matrix
    vector<vector<float>> camera_matrix_transpose;
    for (unsigned int i = 0; i < m_cameraMatrix->capacity(); i++) {
        vector<float> column{};
        for (unsigned int j = 0; j < m_cameraMatrix->at(i).size(); j++) {
            column.push_back(m_cameraMatrix->at(j)[i]);
        }
        camera_matrix_transpose.push_back(column);
    }

    matrix_multiply(result_matrix,camera_matrix_transpose,cuboid_corners);
}

void *PostprocessImageObject6DPoseEstimation::operator()(void           *frameData,
                                                         VecDlTensorPtr &results)
{

    Mat img = Mat(m_config.outDataHeight, m_config.outDataWidth, CV_8UC3, frameData);
    
    void *ret = frameData;
    auto *result = results[0];
    float* data = (float*)result->data;
    int tensorHeight = result->shape[result->dim - 2];
    int tensorWidth = result->shape[result->dim - 1];
#if defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
    DebugDump              &debugObj = getDebugObj();
    string output;
#endif // defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)

    for(int i = 0; i < tensorHeight ; i++)
    {
        vector<int> det_bbox;
        float det_score;
        int det_label;
        vector<float> rotation1;
        vector<float> rotation2;
        vector<vector<float>> rotation_matrix;
        vector<float> translation_vector;

        det_score = data[i * tensorWidth + 4];
        det_label = int(data[i * tensorWidth + 5]);

        if(det_score > m_config.vizThreshold) {
            vector<int> color_map = COLOR_MAP[det_label];
            det_bbox.push_back(data[i * tensorWidth + 0] * m_scaleX);
            det_bbox.push_back(data[i * tensorWidth + 1] * m_scaleY);
            det_bbox.push_back(data[i * tensorWidth + 2] * m_scaleX);
            det_bbox.push_back(data[i * tensorWidth + 3] * m_scaleY);
            rotation1.push_back(data[i * tensorWidth + 6]);
            rotation1.push_back(data[i * tensorWidth + 7]);
            rotation1.push_back(data[i * tensorWidth + 8]);
            rotation2.push_back(data[i * tensorWidth + 9]);
            rotation2.push_back(data[i * tensorWidth + 10]);
            rotation2.push_back(data[i * tensorWidth + 11]);
            translation_vector.push_back(data[i * tensorWidth + 12]);
            translation_vector.push_back(data[i * tensorWidth + 13]);
            translation_vector.push_back(data[i * tensorWidth + 14]);
        
            cross_product_and_transpose(rotation1,rotation2,rotation_matrix);
            vector<vector<float>> vertices = m_vertices[det_label];
            vector<vector<float>> cuboid_corners;
            get_cuboid_corners_2d(vertices,rotation_matrix,translation_vector,m_cameraMatrix,cuboid_corners);

            //Draw Cuboid
            int thickness = 2;
            vector<Point> cuboid_points;
            Scalar color((int)(color_map[0]), (int)(color_map[1]), (int)(color_map[2]));
            for (unsigned int j = 0; j < cuboid_corners.size(); j++)
            {
                int x = (int)(cuboid_corners[j][0]*m_scaleX);
                int y = (int)(cuboid_corners[j][1]*m_scaleY);
                cuboid_points.push_back(Point(x,y));
                circle(img, cuboid_points[j], thickness+3, color, -1);
            }

            //Back
            line(img, cuboid_points[4], cuboid_points[5], color, thickness);
            line(img, cuboid_points[5], cuboid_points[6], color, thickness);
            line(img, cuboid_points[6], cuboid_points[7], color, thickness);
            line(img, cuboid_points[7], cuboid_points[4], color, thickness);

            //Sides
            line(img, cuboid_points[0], cuboid_points[4], color, thickness);
            line(img, cuboid_points[1], cuboid_points[5], color, thickness);
            line(img, cuboid_points[2], cuboid_points[6], color, thickness);
            line(img, cuboid_points[3], cuboid_points[7], color, thickness);

            //Front
            line(img, cuboid_points[0], cuboid_points[1], color, thickness);
            line(img, cuboid_points[1], cuboid_points[2], color, thickness);
            line(img, cuboid_points[2], cuboid_points[3], color, thickness);
            line(img, cuboid_points[3], cuboid_points[0], color, thickness);

            //Put Text along with score
            stringstream ss;
            ss << fixed << setprecision(1) << det_score*100;
            const std::string objectname = m_config.classnames.at(det_label)+ ":" + ss.str() + "%"; 

            Scalar box_color((int)(color_map[0]*0.7), (int)(color_map[1]*0.7), (int)(color_map[2]*0.7));

            Size textSize = getTextSize(objectname, FONT_HERSHEY_SIMPLEX, 0.4, 1, 0); 
            
            Point t_topleft = Point(det_bbox[0],det_bbox[1]);
            Point t_bottomright = Point(det_bbox[0] + textSize.width + 1 , det_bbox[1] + (int)(1.5*textSize.height));
            Point t_text = Point(det_bbox[0] , det_bbox[1] + textSize.height);

            rectangle(img, t_topleft, t_bottomright, box_color, -1);

            int color_map_mean = (color_map[0]+color_map[1]+color_map[2])/3;
            if(color_map_mean > 127)
            {
                Scalar text_color(0,0,0);
                putText(img, objectname, t_text, FONT_HERSHEY_SIMPLEX, 0.4, text_color);
            }
            else
            {
                Scalar text_color(255,255,255);
                putText(img, objectname, t_text, FONT_HERSHEY_SIMPLEX, 0.4, text_color);
            }

#if defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
            output.append(objectname + "[ ");

            for(int32_t j = 0; j < cuboid_points.size(); j++)
            {
                output.append(std::to_string(cuboid_points[j].x) + "," +
                              std::to_string(cuboid_points[j].y) + "," );
            }
            output.append(std::to_string(det_score));
            output.append("]\n");
#endif // defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)


#if defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)
            /* Dump the output object and then increment the frame number. */
            debugObj.logAndAdvanceFrameNum("%s", output.c_str());
#endif // defined(EDGEAI_ENABLE_OUTPUT_FOR_TEST)

        }
    }
    return ret;
}

PostprocessImageObject6DPoseEstimation::~PostprocessImageObject6DPoseEstimation()
{
}

} // namespace ti::edgeai::common