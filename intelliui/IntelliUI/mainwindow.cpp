#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include <QtCharts>
#include <QInputDialog>
#include <IntelliDetect.h>
#include <armadillo>

using namespace QtCharts;
using IntelliDetect::network;
using IntelliDetect::propertyTree;

QString fileName;
network* net = new network(IntelliDetect::sigmoid, IntelliDetect::sigmoidGradient);
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
            QString param_path = QFileDialog::getExistingDirectory(this, tr("Choose location to save data"));
            QString Id = QInputDialog::getText(this,tr("Enter a name for the network"),tr("Name:"));
            propertyTree prop;
            prop.data = string(INTELLI_VERSION);
            prop.setProperty(IntelliDetect::Property::Id,Id.toUtf8().toStdString());
            prop.setProperty(IntelliDetect::Property::hiddenLayerCount,"1");
            prop.setProperty(IntelliDetect::Property::layers::inputLayerSize,"784");
            prop.setProperty(IntelliDetect::Property::layers::hiddenLayerSize(0),"100");
            prop.setProperty(IntelliDetect::Property::layers::outputLayerSize,"10");
            prop.setProperty(IntelliDetect::Property::saveLocation, param_path.toUtf8().toStdString());
            net = new network(prop);
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
    QString net_params = QFileDialog::getExistingDirectory(this, tr("Select network folder"));
    if(!net_params.isEmpty()){
        string param = net_params.toUtf8().constData();
        net = new network(param);
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
    QString Id = QInputDialog::getText(this,tr("Enter a name for the network"),tr("Name:"));
    propertyTree prop;
    prop.data = string(INTELLI_VERSION);
    prop.setProperty(IntelliDetect::Property::Id,Id.toUtf8().toStdString());
    prop.setProperty(IntelliDetect::Property::hiddenLayerCount,"1");
    prop.setProperty(IntelliDetect::Property::layers::inputLayerSize,"784");
    prop.setProperty(IntelliDetect::Property::layers::hiddenLayerSize(0),"100");
    prop.setProperty(IntelliDetect::Property::layers::outputLayerSize,"10");
    net = new network(prop);
}

void MainWindow::on_actionSave_triggered()
{
    QString param_path = QFileDialog::getExistingDirectory(this, tr("Choose location to save data"));
    net->save(param_path.toUtf8().data());
}
