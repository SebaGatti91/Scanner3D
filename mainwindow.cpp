#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <iostream>
#include "qpainter.h"

#include <opencv2/opencv.hpp>   // Include OpenCV AP
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <iostream>

#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>



#include <cmath>



using namespace cv;
using namespace std;
using namespace rs2;

Mat img, depth_image, color_image;
colorizer color_map;
pipeline pip;
pipeline_profile profile;

rs2::config cfg;
rs2::align frames_align(RS2_STREAM_COLOR);
rs2_intrinsics rs2_intr;

rs2::points puntos;
rs2::pointcloud pc;
rs2::frameset aligned_frames;

cv::Mat rvec, tvec;

pcl::PointXYZ local_center_point;

double height_per_pixel;

//nube en pcl punteros para optimizar velocidad

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd_color(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd_color_final(new pcl::PointCloud<pcl::PointXYZRGB>);


//filtros
rs2::threshold_filter th_filter(0.15, 0.4); //filtro de profundidad
rs2::temporal_filter temp_filter; //filtro temporal
rs2::spatial_filter spat_filter; //filtrado espacial conservacion de ejes
rs2::hole_filling_filter hole_filling;// es mejor el de la libreria PCL

frame depth_frames;
frameset frames;


bool is_streaming = false;
bool previous_state = false;
bool color = false;

bool band_final = false;

bool band_primera_captura = false;

int band_depht = 0;
int band_cloud = 0;


double theta; //angulo de rotación
double dtr = CV_PI/180;


//diccionario
std::map<float, int> depth_count;



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{

    ui->setupUi(this);
    timer = new QTimer(this);
    serial = new QSerialPort(); //Inicializa la variable Serial
    arduino_available = false;
    band=0;
    cadena=0;

    foreach (const QSerialPortInfo &serial_Info, QSerialPortInfo::availablePorts()) {//Lee la información de cada puerto serial
        qDebug()<<"Puerto: "<<serial_Info.portName();
        portname = serial_Info.portName();
        qDebug()<<"Vendor Id: "<<serial_Info.vendorIdentifier();
        vendorId = serial_Info.vendorIdentifier();
        qDebug()<<"Product Id: "<<serial_Info.productIdentifier();
        productId = serial_Info.productIdentifier();
        arduino_available = true;
        band=0;
    }

    if(arduino_available){
        arduino_init();
    }

}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::arduino_init()
{
    serial->setPortName(portname);
    serial->setBaudRate(QSerialPort::Baud9600);
    serial->setDataBits(QSerialPort::Data8);
    serial->setParity(QSerialPort::NoParity);
    serial->setStopBits(QSerialPort::OneStop);
    serial->setFlowControl(QSerialPort::NoFlowControl);
    serial->open(QIODevice::ReadWrite);
    connect(serial,SIGNAL(readyRead()),this,SLOT(serial_read()));
}



void MainWindow::serial_read()
{
    if (serial->isWritable() && arduino_available) {
        QByteArray data = serial->readAll();

        // Verificar si se recibió "e" desde Arduino
        if (data.contains("e")) {

            cadena += 60; // Girar 60 grados
            ui->lcdNumber->display(cadena);

            if (cadena == 360) {
                cadena = 0;
                if (serial->isWritable()) {
                    qDebug() << "calculando malla";
                    //serial->write("2"); // Parar la plataforma
                    // mesh(); // Terminar el cálculo de la malla (debes definir esta función)

                    // Realizar un seguimiento del recuento de profundidad en pcd_color_final
                    for (size_t i = 0; i < pcd_color_final->size(); ++i) {
                        float depth_value = pcd_color_final->points[i].z;
                        if (depth_count.find(depth_value) != depth_count.end()) {
                            depth_count[depth_value]++;
                        } else {
                            depth_count[depth_value] = 1;
                        }
                    }

                    // Crear una nueva nube de puntos final que contenga solo los valores de profundidad repetidos más de min_repetitions veces
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_pcd_color(new pcl::PointCloud<pcl::PointXYZRGB>);
                    for (size_t i = 0; i < pcd_color_final->size(); ++i) {
                        float depth_value = pcd_color_final->points[i].z;
                        if (depth_count[depth_value] >= 2) {//repeticiones minimas 2
                            filtered_pcd_color->push_back(pcd_color_final->points[i]);
                        }
                    }


                    pcl::io::savePLYFileASCII("cloud.ply", *pcd_color_final);
                    pcl::io::savePLYFileASCII("FilteredCloud.ply", *filtered_pcd_color);
                    qDebug() << "ply saved";
                   band_final = true;
                }
            }


            if ( !band_final)  {
                pointcloud();
            }
        }
    }
}




void MainWindow::on_pushButton_open_cam_clicked()
{
    is_streaming = true;

    color= true;


    if (is_streaming != previous_state) {

        if (is_streaming)
        {
            connect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
            timer->start(20);
            cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
            cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
            profile = pip.start(cfg);

            //Datos intrinsecos
            auto video_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            rs2_intr = video_profile.get_intrinsics();


            ui->textEdit->setText("Cámara encendida");

        }
        previous_state = is_streaming;
    }
}


void MainWindow::on_pushButton_close_cam_clicked()
{
    is_streaming = false;

    if (is_streaming != previous_state) {

        if (!is_streaming)
        {
            disconnect(timer, SIGNAL(timeout()), this, SLOT(update_window()));

            pip.stop();

            ui->textEdit->setText("Cámara apagada");

            Mat black_img = Mat::zeros(img.size(), CV_8UC3);

            qt_image = QImage((const unsigned char*)(black_img.data), black_img.cols, black_img.rows, QImage::Format_RGB888);

            ui->camera->setPixmap(QPixmap::fromImage(qt_image));

            ui->camera->resize(ui->camera->pixmap()->size());

        }

        previous_state = is_streaming;
    }
}

void MainWindow::update_window()
{

    frames = pip.wait_for_frames(); // Wait for next set of frames from the camera
    auto aligned_frames = frames_align.process(frames);
    rs2::video_frame other_frame = aligned_frames.get_color_frame();
    depth_frames = aligned_frames.get_depth_frame();


    depth_frames = spat_filter.process(depth_frames);//filtrado espacial
    depth_frames = temp_filter.process(depth_frames);// filtrado temporal
    depth_frames = th_filter.process(depth_frames);//filtro distancia

    frame depht_frame_color = depth_frames.apply_filter(color_map);
    img = Mat(Size(640, 480), CV_8UC3, (void*)other_frame.get_data(), Mat::AUTO_STEP);

    if (band_depht==1){
        // Create OpenCV matrix of size (w,h) from the colorized depth data
        img = Mat(Size(640, 480), CV_8UC3, (void*)depht_frame_color.get_data(), Mat::AUTO_STEP);
        //CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
    }

    //convertir a imagen qt
    qt_image = QImage((const unsigned char*)(img.data), img.cols, img.rows, QImage::Format_RGB888);


    //dibujo de líneas
    QPainter painter(&qt_image);
    QPen pen;
    pen.setWidth(2);
    pen.setColor(Qt::red);
    painter.setPen(pen);
    painter.drawLine( min_x,0, min_x,480);
    painter.drawLine( max_x,0, max_x,480);
    painter.drawLine(0,min_y,640,min_y);
    painter.drawLine(0,max_y,640,max_y);    //dibujo de líneas bbox rojo


    // Cambia el color de las líneas a otro color
    QPen centerPen;
    centerPen.setWidth(2);
    centerPen.setColor(Qt::green);  // Cambia Qt::red a Qt::green o al color que desees
    painter.setPen(centerPen);

    // Dibuja las dos últimas líneas con el nuevo color
    painter.drawLine(eje_x_pix, 0, eje_x_pix, 480);
    painter.drawLine(0, eje_y_pix, 640, eje_y_pix);

    //enviar al tablero
    ui->camera->setPixmap(QPixmap::fromImage(qt_image));

}

void MainWindow::on_pushButton_plataforma_on_clicked()
{
    disconnect(timer, SIGNAL(timeout()), this, SLOT(update_window()));

    Mat black_img = Mat::zeros(img.size(), CV_8UC3);

    qt_image = QImage((const unsigned char*)(black_img.data), black_img.cols, black_img.rows, QImage::Format_RGB888);

    ui->camera->setPixmap(QPixmap::fromImage(qt_image));

    qDebug()<<"Encender";

    pointcloud();

}




void MainWindow::on_pushButton_plataforma_off_clicked()
{
    if(serial->isWritable()){
        serial->write("2");
        qDebug()<<"Apagar";
        band = 0;
        cadena = 0;
        band_primera_captura = false;
        ui->lcdNumber->display(cadena);
    }
}



void MainWindow ::pointcloud()

{
    frames = pip.wait_for_frames(); // Wait for next set of frames from the camera
    aligned_frames = frames_align.process(frames);

    rs2::depth_frame depth_frame_pcd = aligned_frames.get_depth_frame();
    rs2::video_frame color_frame_pcd = aligned_frames.get_color_frame();

    depth_frame_pcd = spat_filter.process(depth_frame_pcd);//filtrado espacial
    depth_frame_pcd = th_filter.process(depth_frame_pcd);//filtro distancia

    qDebug()<<"calculando nube";


    //------------------PROCESAMIENTO IMAGEN--------------------------------------------



    // Obtener los datos de profundidad y las coordenadas 3D
    const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_frame_pcd.get_data());
    const uint8_t* color_data = reinterpret_cast<const uint8_t*>(color_frame_pcd.get_data());

    // Find first depth sensor (devices can have zero or more than one)
    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    auto scale =  sensor.get_depth_scale();





    qDebug()<<"calculando nube";

    for (int y = 0; y < rs2_intr.height; ++y) {

        for (int x = 0; x < rs2_intr.width; ++x) {

            float depth_value = depth_data[y * rs2_intr.width + x] * scale;

            //Filtro con límites del bbox

            if (depth_value > 0 && x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
                float px = (x - rs2_intr.ppx) * depth_value / rs2_intr.fx;
                float py = (y - rs2_intr.ppy) * depth_value / rs2_intr.fy;

                float pz = depth_value;

                // Obtener el color correspondiente al punto
                int color_idx = y * rs2_intr.width + x;
                uint8_t r = color_data[color_idx * 3];
                uint8_t g = color_data[color_idx * 3 + 1];
                uint8_t b = color_data[color_idx * 3 + 2];


                pcl::PointXYZ point_depth;

                point_depth.x = px;
                point_depth.y = py;
                point_depth.z = pz;




                //Agrego color a los puntos
                pcl::PointXYZRGB point_color;

                point_color.x = px;
                point_color.y = -py;
                point_color.z = pz;
                point_color.r = r;
                point_color.g = g;
                point_color.b = b;

                pcd_color->push_back(point_color);


                //pcd.points_.push_back(Eigen::Vector3d(px, py, pz));
                //) pcd.colors_.push_back(Eigen::Vector3d(r / 255.0, g / 255.0, b / 255.0));

                // Asignar el color al píxel correspondiente en la imagen
                //image.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
            }
        }
    }


//        // Crea un filtro Passthrough para eliminar puntos por debajo de y = 0.005 metros
//        pcl::PassThrough<pcl::PointXYZRGB> pass;
//        pass.setInputCloud(pcd_color_origen);
//        pass.setFilterFieldName("y");  // Filtrar en la coordenada Z
//        pass.setFilterLimits(-0.005, std::numeric_limits<float>::max());  // Mantener puntos por encima de 0.005 metros
//        pass.filter(*pcd_color_origen);  // Aplicar el filtro



    qDebug()<<"Nube calculada";

    //--------------------------------Rotación respecto eje y-----------------------------





    if (band_primera_captura) {

        //Rotar ángulo eje y
        // Crear una matriz de rotación alrededor del eje Y
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(-cadena * M_PI / 180.0f, Eigen::Vector3f::UnitY()));


        // Aplicar la rotación a la nube de puntos
        pcl::transformPointCloud(*pcd_color, *pcd_color, transform);

        *pcd_color_final += *pcd_color;

    }


    if (!band_primera_captura){
        *pcd_color_final = *pcd_color;
        qDebug()<<"primera captura";
        band_primera_captura = true;
    }




    //indico girar a la plataforma

    pcd_color->clear();

    qDebug()<<"girar plataforma";

    if(serial->isWritable()){

        serial->write("1");

    }
}

void MainWindow::on_pushButton_depht_color_clicked()
{
    if(band_depht==0){
        band_depht =1;
    }
    else {
        band_depht = 0;
    }
}


void MainWindow::mesh(){


    //statistical Outlier Removal Filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(pcd_color_final);
    sor.setMeanK(50); // Número de vecinos para calcular la media y desviación estándar
    sor.setStddevMulThresh(1.0); // Umbral de desviación estándar para determinar valores atípicos

    qDebug()<<"outliers removidos";

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    sor.filter(*cloud_filtered);

    /*
    // Aplicar el filtro Moving Least Squares (MLS)
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
    mls.setInputCloud(cloud_filtered);
    mls.setSearchRadius(0.01); // Radio de búsqueda para vecinos más cercanos
    mls.setPolynomialOrder(2); // Orden del polinomio de ajuste
    mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius(0.005); // Radio de muestreo para el plano local
    mls.setUpsamplingStepSize(0.003); // Tamaño del paso de muestreo

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZRGB>);
    mls.process(*cloud_smoothed);

     qDebug()<<"agujeros rellenados";
*/

    //Guardo nube filtrada
    *pcd_color_final = *cloud_filtered;

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);


    // Calcular Normales
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(cloud_filtered);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);

    /*
    //-----------------------------------CORRECCION DE NORMALES (VER SI SE USA)--------------------------------

    // Aquí deberías iterar a través de los puntos y calcular las direcciones corregidas con ponderación de distancia
    for (size_t i = 0; i < pcd_color->size(); ++i) {
        float x = (*pcd_color)[i].x;
        float y = (*pcd_color)[i].y;|1
        float z = (*pcd_color)[i].z;

        // Calcular dirección desde el centro de rotación al punto
        float direction_x = x - center_x;
        float direction_y = y - center_y;
        float direction_z = z - center_z;

        // Calcular la distancia al centro de rotación
        float distance = sqrt(direction_x * direction_x + direction_y * direction_y + direction_z * direction_z);

        // Aplicar ponderación de distancia a las direcciones
        direction_x *= distance;
        direction_y *= distance;
        direction_z *= distance;

        // Calcular ángulos (por ejemplo, usando atan2) con las direcciones ponderadas
        float angle_x = atan2(direction_y, direction_z); // Ángulo alrededor del eje X
        float angle_y = atan2(direction_z, direction_x); // Ángulo alrededor del eje Y
        float angle_z = atan2(direction_x, direction_y); // Ángulo alrededor del eje Z

        // Corregir la dirección de la normal utilizando los ángulos
        (*normals)[i].normal_x = angle_x;
        (*normals)[i].normal_y = angle_y;
        (*normals)[i].normal_z = angle_z;
    }

    //TODA LA PARTE ANTERIOR SE PUEDE SACAR
    */

    ne.compute(*normals);

    qDebug()<<"normales calculadas";

    // Crear una nube de puntos con normales y colores
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    cloud_with_normals->resize(pcd_color_final->size());

    for (size_t i = 0; i < pcd_color_final->size(); ++i) {
        cloud_with_normals->points[i].x = pcd_color_final->points[i].x;
        cloud_with_normals->points[i].y = pcd_color_final->points[i].y;
        cloud_with_normals->points[i].z = pcd_color_final->points[i].z;
        cloud_with_normals->points[i].normal_x = normals->points[i].normal_x;
        cloud_with_normals->points[i].normal_y = normals->points[i].normal_y;
        cloud_with_normals->points[i].normal_z = normals->points[i].normal_z;
        cloud_with_normals->points[i].r = pcd_color_final->points[i].r;
        cloud_with_normals->points[i].g = pcd_color_final->points[i].g;
        cloud_with_normals->points[i].b = pcd_color_final->points[i].b;
    }

    // Crear un objeto GreedyProjectionTriangulation
    pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
    pcl::PolygonMesh triangles;
    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchRadius(0.025);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 grados
    gp3.setMinimumAngle(M_PI / 18); // 10 grados
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 grados
    gp3.setNormalConsistency(false);

    gp3.reconstruct(triangles);

    // Guardar la malla en formato PLY con normales
    pcl::io::savePLYFile("output_mesh_gp3_with_normals.ply", triangles);

    qDebug()<<"malla lista";

}



void MainWindow::on_pushButton_one_shot_clicked()

{


    disconnect(timer, SIGNAL(timeout()), this, SLOT(update_window()));

    Mat black_img = Mat::zeros(img.size(), CV_8UC3);

    qt_image = QImage((const unsigned char*)(black_img.data), black_img.cols, black_img.rows, QImage::Format_RGB888);

    ui->camera->setPixmap(QPixmap::fromImage(qt_image));


    frames = pip.wait_for_frames(); // Wait for next set of frames from the camera
    aligned_frames = frames_align.process(frames);

    rs2::depth_frame depth_frame_pcd = aligned_frames.get_depth_frame();
    rs2::video_frame color_frame_pcd = aligned_frames.get_color_frame();

    depth_frame_pcd = spat_filter.process(depth_frame_pcd);//filtrado espacial
    depth_frame_pcd = temp_filter.process(depth_frame_pcd);//filtrado temporal
    depth_frame_pcd = th_filter.process(depth_frame_pcd);//filtro distancia




    //------------------PROCESAMIENTO IMAGEN--------------------------------------------



    // Obtener los datos de profundidad y las coordenadas 3D
    const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_frame_pcd.get_data());
    const uint8_t* color_data = reinterpret_cast<const uint8_t*>(color_frame_pcd.get_data());

    // Find first depth sensor (devices can have zero or more than one)
    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    auto scale =  sensor.get_depth_scale();

    //float dist_center = sqrt(pow(local_center_point.z,  2)- pow(0.165 , 2));//165 diff base camara,


    qDebug()<<"calculando nube";

    for (int y = 0; y < rs2_intr.height; ++y) {

        for (int x = 0; x < rs2_intr.width; ++x) {

            float depth_value = depth_data[y * rs2_intr.width + x] * scale;

            //Filtro con límites del bbox

            if (depth_value > 0 && x >= min_x && x <= max_x && y >= min_y && y <= max_y) {

                //correccion parámetros intrínsecos
                float px = (x - rs2_intr.ppx) * depth_value / rs2_intr.fx;
                float py = (y - rs2_intr.ppy) * depth_value / rs2_intr.fy;


//                //proyeccion
//                float pz = 0;

//                float term1 = pow(depth_value, 2);
//                float term2 = pow((165 - (y * height_per_pixel)), 2);

//                if (term1 < 0 || term2 < 0) {
//                    // Manejar caso de valores negativos en pow.
//                    std::cout << "Error: Valores negativos en pow." << std::endl;
//                } else {
//                    pz = sqrt(term2 - term1);
//                    if (std::isnan(pz)) {
//                        // Manejar caso de resultado NaN en sqrt.
//                        std::cout << "Error: Resultado NaN." << std::endl;
//                    } else {
//                        // Hacer algo con el resultado pz.
//                    }
//                }



                // Obtener el color correspondiente al punto
                int color_idx = y * rs2_intr.width + x;
                uint8_t r = color_data[color_idx * 3];
                uint8_t g = color_data[color_idx * 3 + 1];
                uint8_t b = color_data[color_idx * 3 + 2];


                //Agrego color a los puntos
                pcl::PointXYZRGB point_color;

                point_color.x = px-local_center_point.x;
                point_color.y = -py+local_center_point.y;
                point_color.z = -depth_value+local_center_point.z;
                point_color.r = r;
                point_color.g = g;
                point_color.b = b;

                pcd_color->push_back(point_color);


                //pcd.points_.push_back(Eigen::Vector3d(px, py, pz));
                //) pcd.colors_.push_back(Eigen::Vector3d(r / 255.0, g / 255.0, b / 255.0));

                // Asignar el color al píxel correspondiente en la imagen
                //image.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
            }
        }
    }


//    // Crea un filtro Passthrough para eliminar puntos por debajo de Z = 0.005 metros
//    pcl::PassThrough<pcl::PointXYZRGB> pass;
//    pass.setInputCloud(pcd_color_origen);
//    pass.setFilterFieldName("y");  // Filtrar en la coordenada Z
//    pass.setFilterLimits(-0.005, std::numeric_limits<float>::max());  // Mantener puntos por encima de 0.005 metros
//    pass.filter(*pcd_color_origen);  // Aplicar el filtro

    pcl::io::savePLYFileASCII("test.ply", *pcd_color);
    qDebug()<<"ply saved";

    pcd_color->clear();



}


void MainWindow::on_pushButton_detect_base_clicked()

{


    disconnect(timer, SIGNAL(timeout()), this, SLOT(update_window()));

    Mat black_img = Mat::zeros(img.size(), CV_8UC3);

    qt_image = QImage((const unsigned char*)(black_img.data), black_img.cols, black_img.rows, QImage::Format_RGB888);

    ui->camera->setPixmap(QPixmap::fromImage(qt_image));


    frames = pip.wait_for_frames(); // Wait for next set of frames from the camera
    aligned_frames = frames_align.process(frames);

    rs2::depth_frame depth_frame_pcd = aligned_frames.get_depth_frame();
    rs2::video_frame color_frame_pcd = aligned_frames.get_color_frame();

    depth_frame_pcd = spat_filter.process(depth_frame_pcd);//filtrado espacial
    depth_frame_pcd = th_filter.process(depth_frame_pcd);//filtro distancia



    //--------------------DETECCCION PLATAFORMA DE GIRO Y SEGMENTACIÓN-----------------------------

    // Convertir el frame de RealSense a una matriz OpenCV
    cv::Mat rgbImage(cv::Size(color_frame_pcd.as<rs2::video_frame>().get_width(),
                              color_frame_pcd.as<rs2::video_frame>().get_height()),
                     CV_8UC3, (void*)color_frame_pcd.get_data(), cv::Mat::AUTO_STEP);

    // Convertir la imagen de RGB a BGR
    cv::Mat bgrImage;
    cv::cvtColor(rgbImage, bgrImage, cv::COLOR_RGB2BGR);

    // Mostrar la imagen original
    cv::imshow("Imagen", bgrImage);
    cv::waitKey(0);

    // Convertir la imagen de BGR a HSV
    cv::Mat hsvImage;
    cv::cvtColor(bgrImage, hsvImage, cv::COLOR_BGR2HSV);

    // Definir los rangos de color para objetos blancos
    cv::Scalar lowerWhite = cv::Scalar(0, 0, 150); // Rango inferior en HSV
    cv::Scalar upperWhite = cv::Scalar(180, 30, 255); // Rango superior en HSV

    // Filtrar los objetos blancos en el rango especificado
    cv::Mat whiteMask;
    cv::inRange(hsvImage, lowerWhite, upperWhite, whiteMask);

    // Mostrar la máscara de objetos blancos
    cv::imshow("Máscara de Objetos Blancos", whiteMask);
    cv::waitKey(0);


    // Definir el kernel para operaciones de morfología
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // Puedes ajustar el tamaño del kernel

    // Aplicar la operación de apertura (erosión seguida de dilatación) para eliminar el ruido
    cv::Mat denoisedImage;
    cv::morphologyEx(whiteMask, denoisedImage, cv::MORPH_OPEN, kernel);

    // Mostrar la imagen binarizada después de aplicar la operación de apertura
    cv::imshow("Imagen Binarizada Después de la Operación de Apertura", denoisedImage);
    cv::waitKey(0);

    // Encontrar contornos en la imagen binarizada
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(denoisedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    // Filtrar los contornos que podrían ser la base giratoria (por tamaño o forma)
    std::vector<std::vector<cv::Point>> potentialBases;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 12000) {  // Ajusta este umbral según el tamaño esperado de la base
            potentialBases.push_back(contour);
        }
    }

    // Encontrar el contorno con el área más grande entre los contornos potenciales
    double maxArea = 0;
    size_t maxAreaIndex = 0;

    for (size_t i = 0; i < potentialBases.size(); ++i) {
        double area = cv::contourArea(potentialBases[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIndex = i;
        }
    }


    // Obtener los momentos del contorno con el área más grande
    cv::Moments moments = cv::moments(potentialBases[maxAreaIndex]);
    eje_x_pix = moments.m10 / moments.m00;
    eje_y_pix= moments.m01 / moments.m00;

    ui->eje_x->setText(QString::number(eje_x_pix));
    ui->eje_y->setText(QString::number(eje_y_pix));

    // Dibujar el contorno con el área más grande en la imagen resultante
    cv::drawContours(bgrImage, potentialBases, maxAreaIndex, cv::Scalar(0, 255, 0), 2);

    // Dibujar el centro en la imagen resultante
    cv::circle(bgrImage, cv::Point(eje_x_pix, eje_y_pix), 5, cv::Scalar(255, 0, 0), -1);

    // Mostrar la imagen con los contornos potenciales y el centro resaltados
    cv::imshow("Contornos Potenciales y Centro", bgrImage);
    cv::waitKey(0);

    //Centro en coordenadas Locales
    rs2_intrinsics intrinsics = depth_frame_pcd.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

    // Obtener el valor de la profundidad (z) en la imagen de profundidad
    float centerZ = depth_frame_pcd.get_distance(eje_x_pix, eje_y_pix);

    // Calcular las coordenadas en metros en el sistema de la cámara
    float localX = (eje_x_pix - intrinsics.ppx) * centerZ / intrinsics.fx;
    float localY = (eje_y_pix - intrinsics.ppy) * centerZ / intrinsics.fy;

    // Crear un punto XYZ en coordenadas locales
    local_center_point.x = localX;
    local_center_point.y = localY;
    local_center_point.z = centerZ;


    //------------------------------------cálculo de limites de la base-----------------------------------------------

    // Obtener el contorno con el área más grande
    const std::vector<cv::Point>& largestContour = potentialBases[maxAreaIndex];

    // Encontrar los puntos con las coordenadas x mínima y máxima
    int minX = largestContour[0].x;
    int maxX = largestContour[0].x;

    // Encontrar los puntos con las coordenadas y mínima y máxima
    int minY = largestContour[0].y;
    int maxY = largestContour[0].y;

    for (const cv::Point& point : largestContour) {
        if (point.x < minX) {
            minX = point.x;
        }
        if (point.x > maxX) {
            maxX = point.x;
        }
        if (point.y < minY) {
            minY = point.y;
        }
        if (point.y > maxY) {
            maxY = point.y;
        }
    }

    //Asignación de valores

    min_x = minX;
    max_x = maxX;

    min_y = 50;
    max_y = maxY;


}


void MainWindow::on_pushButton_calibrar_ext_clicked()
{


        ui->textEdit->setText("Calculando Matriz");


        // Block program until frames arrive
        frames = pip.wait_for_frames();

        // Try to get a frame of a depth image
        auto aligned_frames = frames_align.process(frames);
        rs2::depth_frame depth_frame= aligned_frames.get_depth_frame();
        rs2::frame color_frame = frames.get_color_frame();


        depth_frame = spat_filter.process(depth_frames);//filtrado espacial
        depth_frame = th_filter.process(depth_frames);//filtro distancia

        // Obtener el perfil de la cámara para acceder a la metadata de calibración
        rs2::video_stream_profile color_stream = color_frame.as<rs2::video_frame>().get_profile().as<rs2::video_stream_profile>();
        rs2_intrinsics intrinsics = color_stream.get_intrinsics();
        rs2_distortion distortion_model = intrinsics.model; //verifico mi modelo de distorción da none

        // Muestra el valor de distortion_model en la consola
        std::cout << "Valor de distortion_model: " << distortion_model << std::endl;


        // Crear la matriz intrínseca de OpenCV
        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        camera_matrix.at<double>(0, 0) = intrinsics.fx;
        camera_matrix.at<double>(1, 1) = intrinsics.fy;
        camera_matrix.at<double>(0, 2) = intrinsics.ppx;
        camera_matrix.at<double>(1, 2) = intrinsics.ppy;

        // Crear la matriz de coeficientes de distorsión de OpenCV
        cv::Mat distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F);

        //las camara realsense poseen el modelo de distorción RS2_DISTORTION_BROWN_CONRADY

        if (distortion_model == RS2_DISTORTION_BROWN_CONRADY) {
            distortion_coefficients.at<double>(0, 0) = intrinsics.coeffs[0]; // k1
            distortion_coefficients.at<double>(0, 1) = intrinsics.coeffs[1]; // k2
            distortion_coefficients.at<double>(0, 2) = intrinsics.coeffs[2]; // p1
            distortion_coefficients.at<double>(0, 3) = intrinsics.coeffs[3]; // p2
            distortion_coefficients.at<double>(0, 4) = intrinsics.coeffs[4]; // k3
        }


        // Convertir el frame de color a una imagen OpenCV
        cv::Mat image(cv::Size(color_frame.as<rs2::video_frame>().get_width(),
                               color_frame.as<rs2::video_frame>().get_height()),
                      CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        // Definir el tamaño del patrón de tablero de ajedrez (número de esquinas)
        cv::Size board_size(6, 4);

        // Encontrar las esquinas del tablero de ajedrez en la imagen RGB
        std::vector<cv::Point2f> image_points;
        bool found = cv::findChessboardCorners(image, board_size, image_points); //image points es un vector que llenaremos con las coordenadas de las esquinas

        if (found) {

            // Dibujar los puntos en la imagen
            cv::drawChessboardCorners(image, board_size, image_points, found);

            // Mostrar la imagen con los puntos
            cv::imshow("Imagen con Puntos de Esquina", image);

            // Esperar a que se presione una tecla (puedes omitir esta línea si no deseas esperar)
            cv::waitKey(0);

            // Mostrar los puntos de esquina por consola
            std::cout << "Puntos de Esquina:" << std::endl;
            for (const auto& point : image_points) {
                std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
            }



            std::vector<cv::Point2f> first_group_points;
            std::vector<cv::Point2f> last_group_points;

            // Separar las coordenadas en dos grupos (primeras 6 y últimas 6)
            for (size_t i = 0; i < image_points.size(); ++i) {
                if (i < 6) {
                    first_group_points.push_back(image_points[i]);
                } else if (i > 18) {
                    last_group_points.push_back(image_points[i]);
                }
            }

            // Calcula el promedio de las alturas de cada grupo
            double sum_height_first_group = 0.0;
            double sum_height_last_group = 0.0;

            for (size_t i = 0; i < first_group_points.size(); ++i) {
                sum_height_first_group += first_group_points[i].y;
                sum_height_last_group += last_group_points[i].y;
            }

            double average_height_first_group = sum_height_first_group / first_group_points.size();
            double average_height_last_group = sum_height_last_group / last_group_points.size();

            double average_height_pixels = average_height_last_group - average_height_first_group;


            //altura real en metros
            double real_height = 84;

            //altura por pixel
             height_per_pixel = real_height / average_height_pixels;



            // Convierte el valor a una cadena de texto (QString)
            QString heightText = QString("Altura promedio en píxeles: %1").arg(average_height_pixels);
            QString realHeightText = QString("Altura real en metros: %1 mm").arg(real_height);
            QString heightPerPixelText = QString("Incremento de altura por píxel: %1 m/píxel").arg(height_per_pixel);

            // Muestra ambos valores en el widget de texto
               ui->textEdit->setText(realHeightText + "\n" + heightText+ "\n" + heightPerPixelText );
        }

        else {
            ui->textEdit->setText("Error al buscar esquinas");
        }

   //----------------------------------------------------------------------------------------------------------------------------

//    // Definir puntos de correspondencia en la imagen y en el mundo real
//    std::vector<cv::Point2f> puntosImagen;
//    std::vector<cv::Point2f> puntosMundoReal;

//    // Rellenar los puntos de correspondencia, por ejemplo:
//    puntosImagen.push_back(cv::Point2f(10, 20));
//    puntosImagen.push_back(cv::Point2f(300, 50));
//    puntosImagen.push_back(cv::Point2f(450, 400));
//    puntosImagen.push_back(cv::Point2f(50, 400));

//    puntosMundoReal.push_back(cv::Point2f(0, 0));
//    puntosMundoReal.push_back(cv::Point2f(1, 0));
//    puntosMundoReal.push_back(cv::Point2f(1, 1));
//    puntosMundoReal.push_back(cv::Point2f(0, 1));

//    // Calcular la matriz de transformación de perspectiva
//    cv::Mat matrizTransformacion = cv::getPerspectiveTransform(puntosImagen, puntosMundoReal);

//    // Simular datos de profundidad como una nube de puntos en 3D
//    std::vector<cv::Point3f> nubePuntos3D;
//    // Rellenar tu nube de puntos en 3D con coordenadas (x, y, z) y datos de profundidad

//    for (cv::Point3f &punto : nubePuntos3D) {
//        // Convertir cada punto utilizando la matriz de transformación y la profundidad
//        cv::Mat puntoMatriz = (cv::Mat_<float>(3, 1) << punto.x, punto.y, punto.z);
//        cv::Mat puntoTransformado = matrizTransformacion * puntoMatriz;

//        // Normalizar por la coordenada de profundidad
//        float profundidad = punto.z;
//        puntoTransformado /= profundidad;

//        punto.x = puntoTransformado.at<float>(0, 0);
//        punto.y = puntoTransformado.at<float>(1, 0);
//        punto.z = puntoTransformado.at<float>(2, 0);
//    }

//    // Ahora, nubePuntos3D contiene tus puntos transformados



}

void MainWindow::on_pushButton_centro_clicked()
{
    frames = pip.wait_for_frames(); // Wait for next set of frames from the camera
    aligned_frames = frames_align.process(frames);

    rs2::depth_frame depth_frame_pcd = aligned_frames.get_depth_frame();
    rs2::video_frame color_frame_pcd = aligned_frames.get_color_frame();

    depth_frame_pcd = spat_filter.process(depth_frame_pcd);//filtrado espacial
    depth_frame_pcd = th_filter.process(depth_frame_pcd);//filtro distancia

    eje_x_pix = ui->eje_x->text().toDouble();
    eje_y_pix = ui->eje_y->text().toDouble();

    //Centro en coordenadas Locales
    rs2_intrinsics intrinsics = depth_frame_pcd.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

    // Obtener el valor de la profundidad (z) en la imagen de profundidad
    float centerZ = depth_frame_pcd.get_distance(eje_x_pix , eje_y_pix);

    // Calcular las coordenadas en metros en el sistema de la cámara
    float localX = (eje_x_pix  - intrinsics.ppx) * centerZ / intrinsics.fx;
    float localY = (eje_y_pix - intrinsics.ppy) * centerZ / intrinsics.fy;


    // Crear un punto XYZ en coordenadas locales
    local_center_point.x = localX;
    local_center_point.y = localY;
    local_center_point.z = centerZ;

}

