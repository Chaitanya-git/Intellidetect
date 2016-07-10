/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionTrain_from_file;
    QAction *actionTrain_for_current_input;
    QAction *actionLoad_network_from_file;
    QAction *actionView_training_statistics;
    QWidget *centralWidget;
    QLabel *result_label;
    QLabel *confidence_label;
    QPushButton *load_btn;
    QPushButton *process_btn;
    QLabel *pic_label;
    QLabel *result_val_label;
    QLabel *confidence_val_label;
    QMenuBar *menuBar;
    QMenu *menuOptions;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1024, 768);
        actionTrain_from_file = new QAction(MainWindow);
        actionTrain_from_file->setObjectName(QStringLiteral("actionTrain_from_file"));
        actionTrain_from_file->setMenuRole(QAction::ApplicationSpecificRole);
        actionTrain_for_current_input = new QAction(MainWindow);
        actionTrain_for_current_input->setObjectName(QStringLiteral("actionTrain_for_current_input"));
        actionTrain_for_current_input->setMenuRole(QAction::ApplicationSpecificRole);
        actionLoad_network_from_file = new QAction(MainWindow);
        actionLoad_network_from_file->setObjectName(QStringLiteral("actionLoad_network_from_file"));
        actionLoad_network_from_file->setShortcutContext(Qt::ApplicationShortcut);
        actionView_training_statistics = new QAction(MainWindow);
        actionView_training_statistics->setObjectName(QStringLiteral("actionView_training_statistics"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        result_label = new QLabel(centralWidget);
        result_label->setObjectName(QStringLiteral("result_label"));
        result_label->setGeometry(QRect(30, 560, 141, 61));
        QFont font;
        font.setPointSize(15);
        result_label->setFont(font);
        confidence_label = new QLabel(centralWidget);
        confidence_label->setObjectName(QStringLiteral("confidence_label"));
        confidence_label->setGeometry(QRect(30, 600, 141, 61));
        confidence_label->setFont(font);
        load_btn = new QPushButton(centralWidget);
        load_btn->setObjectName(QStringLiteral("load_btn"));
        load_btn->setGeometry(QRect(50, 40, 161, 61));
        process_btn = new QPushButton(centralWidget);
        process_btn->setObjectName(QStringLiteral("process_btn"));
        process_btn->setGeometry(QRect(730, 40, 161, 61));
        pic_label = new QLabel(centralWidget);
        pic_label->setObjectName(QStringLiteral("pic_label"));
        pic_label->setGeometry(QRect(50, 130, 491, 421));
        result_val_label = new QLabel(centralWidget);
        result_val_label->setObjectName(QStringLiteral("result_val_label"));
        result_val_label->setGeometry(QRect(233, 580, 71, 31));
        confidence_val_label = new QLabel(centralWidget);
        confidence_val_label->setObjectName(QStringLiteral("confidence_val_label"));
        confidence_val_label->setGeometry(QRect(230, 610, 71, 31));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1024, 22));
        menuOptions = new QMenu(menuBar);
        menuOptions->setObjectName(QStringLiteral("menuOptions"));
        MainWindow->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuOptions->menuAction());
        menuOptions->addAction(actionLoad_network_from_file);
        menuOptions->addAction(actionTrain_from_file);
        menuOptions->addAction(actionTrain_for_current_input);
        menuOptions->addAction(actionView_training_statistics);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "IntelliDetect", 0));
        actionTrain_from_file->setText(QApplication::translate("MainWindow", "&Train from file", 0));
        actionTrain_for_current_input->setText(QApplication::translate("MainWindow", "Train &for current input", 0));
        actionTrain_for_current_input->setShortcut(QApplication::translate("MainWindow", "Ctrl+T", 0));
        actionLoad_network_from_file->setText(QApplication::translate("MainWindow", "Load network from file", 0));
        actionLoad_network_from_file->setShortcut(QApplication::translate("MainWindow", "Ctrl+L", 0));
        actionView_training_statistics->setText(QApplication::translate("MainWindow", "View training statistics", 0));
        result_label->setText(QApplication::translate("MainWindow", "Result : ", 0));
        confidence_label->setText(QApplication::translate("MainWindow", "Confidence : ", 0));
        load_btn->setText(QApplication::translate("MainWindow", "Load Image", 0));
        process_btn->setText(QApplication::translate("MainWindow", "Process Image", 0));
        pic_label->setText(QString());
        result_val_label->setText(QString());
        confidence_val_label->setText(QString());
        menuOptions->setTitle(QApplication::translate("MainWindow", "Tools", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
