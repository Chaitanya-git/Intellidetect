/*
 * Copyright (C) 2017 Chaitanya and Geeve George
 * This file is part of Intellidetect.
 *
 *  Intellidetect is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Intellidetect is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Intellidetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef PROPTREE_H
#define PROPTREE_H

#include <constants.h>

struct propertyTree{
        string data;
        vector< pair<string, propertyTree> > childNodes;
    public:
        bool load(string);
        string getProperty(string);
        void setProperty(string,string);
        void setPropertyIfNotSet(string,string);
        bool isSet(string);
        string toString(int);
};

bool propertyTree::load(string path){
    childNodes.clear();
    fstream prop;
    prop.open(path,ios::in);
    if(!prop)
        return false;
    cout<<"\nopened file"<<endl;
    string line;
    vector<string> prevProps;
    unsigned int level = 0;
    do{
        getline(prop, line);
        if(line[0]=='/' && line[1] == '/')
            continue;
        else if(line[0] == '\t' || line[0] == ' ')
            ++level;
        string propName,value;
        bool flag = false;
        for(unsigned int i=0;i<line.length();++i){
            if(line[i]==' ') continue;
            if(line[i]== '='){
                flag = true;
                continue;
            }
            if(flag)
                value.append(&line[i],1);
            else if(line[i]!='\t')
                propName.append(&line[i],1);
        }
        if(!value.length())
            if(level>=prevProps.size())
                prevProps.push_back(propName);
            else
                prevProps[level] = propName;
        else{
            for(unsigned int i=level-1;i<prevProps.size();--i)
                propName = (prevProps[i]+string(".")+propName);
            if(!propName.compare("version"))
                data = value;
            else
                setProperty(propName,value);
            level=0;
        }
    }while(!prop.eof());
    prop.close();
    return true;
}
void propertyTree::setProperty(string property, string value){
    bool flag=false;
    string nodeName(""),propName("");
    for(unsigned int i=0;i<property.length();++i){
        if(property[i]=='.'){
            flag=true;
            ++i;
        }
        if(flag)
            propName.append(&property[i],1);
        else
            nodeName.append(&property[i],1);
    }
    for(unsigned int i=0;i<childNodes.size();++i){
        if(!get<0>(childNodes[i]).compare(nodeName)){
            if(propName.length())
                get<1>(childNodes[i]).setProperty(propName,value);
            else
                get<1>(childNodes[i]).data = value;
            return;
        }
    }
    childNodes.push_back(make_pair(nodeName,*(new propertyTree)));
    if(propName.length())
        get<1>(childNodes.back()).setProperty(propName,value);
    else
        get<1>(childNodes.back()).data = value;
}

void propertyTree::setPropertyIfNotSet(string property, string value){
    if(!isSet(property))
        setProperty(property,value);
}

string propertyTree::getProperty(string property){
    bool flag=false;
    string nodeName,propName;
    for(unsigned int i=0;i<property.length();++i){
        if(property[i]=='.'){
            flag=true;
            ++i;
        }
        if(flag)
            propName.append(&property[i],1);
        else
            nodeName.append(&property[i],1);
    }
    for(auto prop: childNodes){
        if(!get<0>(prop).compare(nodeName)){
            return get<1>(prop).getProperty(propName);

        }
    }
    return data;
}

bool propertyTree::isSet(string property){
    bool flag=false;
    string nodeName,propName;
    for(unsigned int i=0;i<property.length();++i){
        if(property[i]=='.'){
            flag=true;
            ++i;
        }
        if(flag)
            propName.append(&property[i],1);
        else
            nodeName.append(&property[i],1);
    }
    for(auto prop: childNodes){
        if(!get<0>(prop).compare(nodeName)){
            if(propName.length())
                return get<1>(prop).isSet(propName);
            else
                return true;

        }
    }
    return false;
}

string propertyTree::toString(int level=0){
    string strRep("");
    if(!level){
        strRep+=string("version = ");
    }
    strRep+=(data+string("\n"));
    for(auto prop: childNodes){
        for(int i=0;i<level;++i)
            strRep+=string("\t");
        strRep+=(get<0>(prop) + string(" = ") + get<1>(prop).toString(level+1));
    }
    return strRep;
}

#endif // PROPTREE_H
