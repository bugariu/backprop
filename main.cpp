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

#include "network.h"

int main(int argc, char *argv[])
{
    Q_UNUSED(argc);
    Q_UNUSED(argv);

    qsrand(time(0));
    backprop::Network network{{2, 4, 1}};

    double gamma = 0.3;
    double dd = 0.999;
    for(int i=0; i<100000; i++)
    {
        auto result = network.FeedForward({0, 0});
        qInfo() << result;
        network.BackPropagation({0}, gamma);
        result = network.FeedForward({0, 1});
        qInfo() << result;
        network.BackPropagation({1}, gamma);
        result = network.FeedForward({1, 0});
        qInfo() << result;
        network.BackPropagation({1}, gamma);
        result = network.FeedForward({1, 1});
        qInfo() << result;
        network.BackPropagation({0}, gamma);
        qInfo() << "---";
        gamma = gamma*dd;
    }
}
