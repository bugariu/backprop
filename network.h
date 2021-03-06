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
#include <cmath>

namespace backprop
{

/**
 * @todo comment
 */
class Network
{

private:

    /**
     * @todo comment
     */
    class Neuron
    {
    public:
        double              m_Output{0};    ///< @todo comment
        double              m_Net{0};       ///< @todo comment
        double              m_Error{0};     ///< @todo comment
        QVector<Neuron *>   m_Predecessors; ///< @todo comment
        QVector<double>     m_Weights;      ///< @todo comment
        QVector<double>     m_NewWeights;   ///< @todo comment
        Neuron() = default;
        /**
         * @todo comment
         */
        Neuron(const QVector<Neuron *> &predecessors):
            m_Predecessors(predecessors)
        {
            m_NewWeights.resize(predecessors.size());
            m_Weights.resize(predecessors.size());
            for(auto & it : m_NewWeights)
            {
                it = 5*(RAND_MAX -2.0*rand())/RAND_MAX;
            }
            SetNewWeights();
        }
        /**
         * @todo comment
         */
        void FeedForward()
        {
            m_Error = 0;
            m_Net = 0;
            for(int i = 0; i< m_Weights.size(); i++)
            {
                m_Net += m_Weights[i]*m_Predecessors[i]->m_Output;
            }
            m_Output = f(m_Net);
        }
        /**
         * @todo comment
         */
        void BackPropagation(double outputError, double gamma)
        {
            m_Error = df(m_Net)*outputError;
            BackPropAdjust(m_Error, gamma);
        }
        /**
         * @todo comment
         */
        void SetNewWeights()
        {
            m_Weights = m_NewWeights;
        }
        /**
         * @todo comment
         */
        double WeightedError(int idx)
        {
            return m_Error*m_Weights[idx];
        }
    private:
        /**
         * @todo comment
         */
        double f(double x)
        {
            return 1.0/(1+exp(-x));
        }
        /**
         * @todo comment
         */
        double df(double x)
        {
            auto tmp = f(x);
            return tmp*(1-tmp);
        }
        /**
         * @todo comment
         */
        void BackPropAdjust(double delta, double gamma)
        {
            for(int i=0; i< m_NewWeights.size(); i++)
            {
                m_NewWeights[i] = m_Weights[i] + gamma*delta*m_Predecessors[i]->m_Output;
            }
        }
    };
public:
    /**
     * @todo comment
     */
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
        m_Layers.resize(layerSizes.size()-1);
        for(int i=0; i<layerSizes.size()-1; i++)
        {
            /// @todo check for at leas one neuron.
            QVector<Neuron *> layer;
            layer.resize(layerSizes[i+1]);
            for(auto & it : layer)
            {
                if(i == 0)
                {
                    // first layer
                    it = new Neuron{m_InputLayer};
                }
                else
                {
                    it = new Neuron{m_Layers[i-1]};
                }
            }
            m_Layers[i] = layer;
        }
    }
    /**
     * @todo comment
     */
    ~Network()
    {
        qDeleteAll(m_InputLayer);
        for(auto & it : m_Layers)
        {
            qDeleteAll(it);
        }
    }
    /**
     * @todo comment
     */
    QVector<double> FeedForward(const QVector<double> &inputValues)
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
    /**
     * @todo comment
     */
    void BackPropagation(const QVector<double> &outputValues, double gamma)
    {
        /// @todo check number of outputs == output layer size
        for(int i=0; i<m_Layers.size(); i++)
        {
            auto layer = m_Layers[m_Layers.size() - i - 1];
            if(i==0)
            {
                // last layer
                for(int n = 0; n < layer.size(); n++)
                {
                    layer[n]->BackPropagation(outputValues[n]-layer[n]->m_Output, gamma);
                }
            }
            else
            {
                // not last layer
                auto successor = m_Layers[m_Layers.size() - i];
                for(int n = 0; n < layer.size(); n++)
                {
                    double sum = 0;
                    for(int k=0; k<successor.size(); k++)
                    {
                        sum += successor[k]->WeightedError(n);
                    }
                    layer[n]->BackPropagation(sum, gamma);
                }
            }
        }
        /**
         * @todo comment
         */
        for(auto & it1 : m_Layers)
        {
            for(auto & it2 : it1)
            {
                it2->SetNewWeights();
            }
        }
    }

private:
    QVector<QVector<Neuron *>>  m_Layers;       ///< @todo comment
    QVector<Neuron *>           m_InputLayer;   ///< @todo comment
};

} // namespace backprop

#endif // BACKPROP_NETWORK_H
