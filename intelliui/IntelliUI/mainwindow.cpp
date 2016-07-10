#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include <QInputDialog>
#include <IntelliDetect.h>
#include <armadillo>

QString fileName;
ReLU* net = new ReLU();
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

void MainWindow::on_actionTrain_from_file_triggered()
{
    QFileDialog dialog(this);
    dialog.setNameFilter(tr("CSV Data files (*.csv)"));
    dialog.setViewMode(QFileDialog::Detail);
    QStringList csv_files = QFileDialog::getOpenFileNames(this, tr("Open File"),"",tr("CSV Files (*.csv)"));
    vector<string> fileNames;
    fileNames.resize(0);
    for(int i=0;i<csv_files.size();++i){
        fileNames.push_back(csv_files.at(i).toUtf8().constData());
    }
    if(!csv_files.isEmpty()){
        net->load(fileNames);
        net->train();
    }
}

void MainWindow::on_actionTrain_for_current_input_triggered()
{
    string fileStr = fileName.toUtf8().constData();
    int Label = QInputDialog::getInt(this,tr("Enter Label"),tr("Digit Label: "),0,0,9);
//    cout<<"Size of X = "<<X_batch.n_rows<<"x"<<X_batch.n_cols;
//    cout<<"Size of Y = "<<Y_batch.n_rows<<"x"<<Y_batch.n_cols;
//    net->load(X_batch, Y_batch);
    net->train(fileStr, Label, 0.0,0.0001);
    //net->train(1, fileStr);
}
