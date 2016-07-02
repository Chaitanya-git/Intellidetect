#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include "/home/chaitanya/Downloads/IntelliDetect 2.0/IntelliDetect.h"
#include <armadillo>

QString fileName;
network* net = new network();
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_load_btn_clicked()
{
    QFileDialog dialog(this);
        dialog.setNameFilter(tr("Images (*.pgm)"));
        dialog.setViewMode(QFileDialog::Detail);
        fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                        "",
                                                        tr("Images (*.pgm)"));


        QPixmap imgpixmap(fileName);

        ui->pic_label->setPixmap(imgpixmap);
        ui->pic_label->setScaledContents( true );

        ui->pic_label->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );


}
void MainWindow::on_process_btn_clicked()
{
    string fileStr = fileName.toUtf8().constData();
    int output = as_scalar(net->predict(fileStr));
    float confidence = as_scalar(net->output(fileStr));
    ui->result_val_label->setText(QString::number(output));
    ui->confidence_val_label->setText(QString::number(confidence*100));
}

void MainWindow::on_trainButton_clicked()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("CSV Data files (*.csv)"));
    dialog.setViewMode(QFileDialog::Detail);
    QString csv_file = QFileDialog::getOpenFileName(this, tr("Open File"),"",tr("CSV Files (*.csv)"));
    if(!csv_file.isEmpty()){
        net->load(csv_file.toUtf8().constData());
        net->train();
    }
}
