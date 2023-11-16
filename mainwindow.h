#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QTimer>
#include <QMainWindow>
#include <QtSerialPort/qserialport.h>
#include <QtSerialPort/qserialportinfo.h>
#include <QDebug>
#include <QWindow>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    void on_pushButton_open_cam_clicked();

    void on_pushButton_close_cam_clicked();

    void update_window();

    void arduino_init();

    void serial_read();

    void mesh();

    void pointcloud();

    void on_pushButton_depht_color_clicked();

    void on_pushButton_plataforma_on_clicked();

    void on_pushButton_plataforma_off_clicked();

    void on_pushButton_one_shot_clicked();

    void on_pushButton_centro_clicked();

    void on_pushButton_detect_base_clicked();

    void on_pushButton_calibrar_ext_clicked();



private:

    Ui::MainWindow *ui;
    QTimer* timer;
    QSerialPort *serial;
    QImage qt_image;
    QString portname;
    quint16 band = 0;
    quint16 vendorId;
    quint16 productId;
    double cadena = 0;
    bool arduino_available;
    float min_x, min_y, max_x, max_y, eje_x_pix, eje_y_pix,center_x,center_y,center_z;


};
#endif // MAINWINDOW_H
