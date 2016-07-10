#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include <QtCharts>
#include <QInputDialog>
#include <IntelliDetect.h>
#include <armadillo>

using namespace QtCharts;

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
    QStringList csv_files = QFileDialog::getOpenFileNames(this, tr("Select training sets"),"",tr("CSV Files (*.csv)"));
    QString test_set = QFileDialog::getOpenFileName(this, tr("Select test set"),"",tr("CSV Files (*.csv)"));
    vector<string> fileNames;
    fileNames.resize(0);
    for(int i=0;i<csv_files.size();++i){
        fileNames.push_back(csv_files.at(i).toUtf8().constData());
    }
    fileNames.push_back(test_set.toUtf8().constData());
    if(!csv_files.isEmpty()){
        if(net->getPath().empty()){
            QString param_path = QFileDialog::getSaveFileName(this, tr("Choose file to save data"),"",tr("Network Parameter files (*.CSV)"));
            net = new ReLU(param_path.toUtf8().constData());
        }
        net->load(fileNames);
        net->train();
    }
}

void MainWindow::on_actionTrain_for_current_input_triggered()
{
    string fileStr = fileName.toUtf8().constData();
    int Label = QInputDialog::getInt(this,tr("Enter Label"),tr("Digit Label: "),0,0,9);
    net->train(fileStr, Label, 0.0,0.0001);
}

void MainWindow::on_actionLoad_network_from_file_triggered()
{
    QString net_params = QFileDialog::getOpenFileName(this, tr("Select network file"),"",tr("Network parameter files (*.CSV)"));
    if(!net_params.isEmpty()){
        string param = net_params.toUtf8().constData();
        net = new ReLU(param);
    }
}

void MainWindow::on_actionView_training_statistics_triggered()
{
    if(!net->trainSetCostsReg.size()){
        QMessageBox noStatMsg;
        noStatMsg.setText("No statistics available");
        noStatMsg.exec();
        return;
    }
    QLineSeries *series = new QLineSeries();
    for(unsigned int i=0;i<net->trainSetCostsReg.size();++i){
        *series << QPointF(i+1,net->trainSetCostsReg.at(i));
    }

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Cost vs Number of Iterations");
    chart->createDefaultAxes();
    chart->axisX()->setRange(0,(unsigned long long)(net->trainSetCostsReg.size()+10));
    chart->axisY()->setRange(0,(unsigned long long)(*(std::max_element(net->trainSetCostsReg.begin(),net->trainSetCostsReg.end()))));
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->resize(1024,768);
    chartView->show();
}

void MainWindow::on_actionNew_network_triggered()
{
    net = new ReLU();
}
