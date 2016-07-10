#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_load_btn_clicked();
    void on_process_btn_clicked();
    void on_actionTrain_from_file_triggered();
    void on_actionTrain_for_current_input_triggered();

    void on_actionLoad_network_from_file_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
