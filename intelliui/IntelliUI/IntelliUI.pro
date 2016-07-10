#-------------------------------------------------
#
# Project created by QtCreator 2016-06-01T08:42:22
#
#-------------------------------------------------

QT       += core gui
QT       += charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = IntelliUI
TEMPLATE = app


QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -O3
QMAKE_CXXFLAGS += -larmadillo
QMAKE_CXXFLAGS += -llapack
QMAKE_CXXFLAGS += -lopenblas
LIBS += -llapack
LIBS += -lblas
LIBS += -larmadillo

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h
INCLUDEPATH += "/home/chaitanya/IntelliDetect 2.1/"
FORMS    += mainwindow.ui

