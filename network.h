/******************************************************************************//**
 * @file
 * @brief Main routine
 *
 * @copyright Copyright (C) by Doru Julian Bugariu
 * julian@bugariu.eu
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the
 * Free Software Foundation
 * 51 Franklin Street, Fifth Floor
 * Boston, MA 02110-1301, USA
 * http://www.fsf.org/about/contact.html
 *********************************************************************************/

#ifndef BACKPROP_NETWORK_H
#define BACKPROP_NETWORK_H

#include <QVector>
#include <QDebug>
#include <cmath>

namespace backprop
{

class Network
{
private:
    class Neuron
    {
    public:
        double              m_Output{0};
        QVector<Neuron *>   m_Predecessors;
        QVector<double>     m_Weights;
        Neuron() = default;
        Neuron(const QVector<Neuron *> &predecessors):
            m_Predecessors(predecessors)
        {
            m_Weights.resize(predecessors.size());
            for(auto & it : m_Weights)
            {
                it = 1.0*(rand() - RAND_MAX/2)/RAND_MAX;
                qInfo() << it;
            }
        }
        void FeedForward()
        {
            m_Output = 0;
            for(int i = 0; i< m_Weights.size(); i++)
            {
                m_Output += m_Weights[i]*m_Predecessors[i]->m_Output;
            }
            m_Output = f(m_Output);
        }
    private:
        double f(double x)
        {
            return 1.0/(1+exp(x));
        }
    };
public:
    Network(const QVector<int> &layerSizes)
    {
        /// @todo check for at leas one layer.
        // create input layer
        int inputLayerSize = layerSizes[0];
        m_InputLayer.resize(inputLayerSize);
        for (auto & it : m_InputLayer)
        {
            it = new Neuron;
        }
        // create layers
        m_Layers.resize(layerSizes.size());
        for(int i=1; i<layerSizes.size(); i++)
        {
            /// @todo check for at leas one neuron.
            QVector<Neuron *> layer;
            layer.resize(layerSizes[i]);
            for(auto & it : layer)
            {
                if(i == 1)
                {
                    // first layer
                    it = new Neuron{m_InputLayer};
                }
                else
                {
                    // not first layer
                    it = new Neuron{m_Layers[i-1]};
                }
            }
            m_Layers[i] = layer;
        }
    }
    ~Network()
    {
        qDeleteAll(m_InputLayer);
        for(auto & it : m_Layers)
        {
            qDeleteAll(it);
        }
    }
    QVector<double> FeedForward(QVector<double> inputValues)
    {
        /// @todo check number of inputs == input layer size
        // set input layer
        for(int i=0; i<inputValues.size(); i++)
        {
            m_InputLayer[i]->m_Output = inputValues[i];
        }
        // iterate through layers and compute output
        for(auto & layer : m_Layers)
        {
            for(auto & neuron : layer)
            {
                neuron->FeedForward();
            }
        }
        // set result
        QVector<double> result;
        result.resize(m_Layers.last().size());
        for(int i=0; i<result.size(); i++)
        {
            result[i] = m_Layers.last()[i]->m_Output;
        }
        return result;
    }

private:
    QVector<QVector<Neuron *>>  m_Layers;
    QVector<Neuron *>           m_InputLayer;
};

} // namespace backprop

#endif // BACKPROP_NETWORK_H
