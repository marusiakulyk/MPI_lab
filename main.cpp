Marusia Kulyk, [27.05.19 17:11]
#include<stdio.h>
#include<mpi.h>
#include <functional>
#include <cmath>
#include <future>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include <cerrno>
#include <algorithm>
#include <iostream>

template <class T>
int check_parameters(std::map<std::string, T> config, std::vector<std::string>* required_args){
    for(int i = 0, end = (int)required_args->size(); i < end; i++){
        if (config.find(required_args->at(i)) == config.end()) {
            errno = EINVAL;
            return i;
        }
        i++;
    }
    return -1;
}

template <class T>
std::map<std::string, T> read_config_from_file(const std::string& filename){
    std::map <std::string, T> config;

    std::ifstream file (filename);

    if (file.is_open())
    {
        try {
            std::string line;
            while(getline(file, line)){
                line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
                if(line.empty() || line[0] == '#')
                    continue;
                auto delimiter = line.find('=');

                auto name = line.substr(0, delimiter);
                auto value = line.substr(delimiter + 1);
                value.erase(std::remove_if(value.begin(), value.end(), [](auto x){return x == '"';}), value.end());
                config.insert({name, std::stod(value)});
            }
        } catch (const std::invalid_argument& ia) {
            errno = EINVAL;
        }
    }
    else {
        errno = ENOENT;
    }

    return config;

}


const auto function = [](double x, double y){
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 1; i <= 5; ++i){
        sum1 += i * cos((i + 1) * x + 1);
        sum2 += i * cos((i + 1) * y + 1);
    }
    return - sum1 * sum2;
};

void integral(double start_x, double finish_x, double start_y, double finish_y,
              const std::function<double(double, double)> &function, double step,
              std::promise<double> && promise){

    double result = 0;
    for(double x = start_x; x <= finish_x; x += step){
        for(double y = start_y; y < finish_y; y += step){
            result += function(x - step/2, y - step/2) * step * step;
        }
    }
    promise.set_value(result);
}


double integral_threads(double start_x, double finish_x, double start_y, double finish_y,
                                                int num_steps, int num_threads){
    double result = 0;
    double step_y = floor((finish_y - start_y)/num_threads);
    double upper = finish_y + floor((finish_y - start_y)/num_steps);
    std::vector<std::thread> threads;
    std::vector<std::future<double>> f;

    for(int i = 0; i < num_threads; i++){
        std::promise<double> p;
        f.push_back(p.get_future());
        threads.emplace_back(integral, start_x, finish_x, start_y + i * step_y,
                             (i != num_threads - 1) ? start_y + (i + 1) * step_y : upper,
                             function, (finish_x - start_x)/ ((double)num_steps), std::move(p));
    }

    for (int i = 0; i < num_threads; i++){
        result += f[i].get();
    }
    for(auto& t: threads) {t.join();}
    return result;
}

double one_process(double abs, double rel, int num_threads, double start_x, double finish_x, double start_y, double finish_y ){
    int max_iterations = 5;
    double  previous_res = 0,
            current_res = 0;
    int num_steps = 200;


    while(max_iterations){
        std::cout << "curr_iter: " << max_iterations << '\n';
        previous_res = current_res;
        current_res = integral_threads(start_x, finish_x,start_y,
                                       finish_y, num_steps, num_threads);

        if (previous_res != 0) {
            if (fabs(current_res - previous_res) <= abs &&
                fabs((current_res - previous_res) / previous_res) <= rel) {
                break;
            }
        }

        num_steps *= 2;
        max_iterations --;
    }
    return current_res;
}
int main(int argc, char *argv []){
    int commsize, rank, len ;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(procname, &len);
    if(rank==0) {
        std::string config_file;

        if (argc < 2) {
            std::cerr << "No config!\n";
            MPI_Finalize();
        } else {
            config_file = argv[1];
        }


        std::vector<std::string> required_args{"absolute_error",
                                               "relative_error",
                                               "num_of_threads",
                                               "start_x",
                                               "finish_x",
                                               "start_y",
                                               "finish_y"};

        std::map<std::string, double> config{read_config_from_file<double>(config_file)};

        int index = check_parameters(config, &required_args);
        if (index != -1) {
            std::cerr << "You did not specify " << required_args[index];
            MPI_Finalize();
        }
        long long steps = 200;
        double stepx = (config["finish_x"] - config["start_x"]) / steps;
        double sbuff[] = {config["absolute_error"], config["relative_error"],
                          config["num_of_threads"], config["start_x"],
                          config["start_x"] + stepx, config["start_y"], config["finish_y"]};
        for (int i = 1; i < commsize; i++) {
            MPI_Send(sbuff, 7, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            sbuff[3] = sbuff[4];
            sbuff[4] += stepx;
        }

        double res = one_process(sbuff[0], sbuff[1], sbuff[2], sbuff[4], sbuff[4] + stepx, sbuff[5], sbuff[6]);
        double send_buf[] = {res};
        double *rbuff = (double *) malloc(sizeof(double));

        MPI_Reduce((const void *) send_buf, (void *) rbuff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        std::cout<<"result: " << rbuff[0];
    }
        else{
            double *rbuf;
            MPI_Recv(rbuf, 7, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double result = one_process(rbuf[0], rbuf[1], rbuf[2], rbuf[3], rbuf[4], rbuf[5], rbuf[6]);
        double *rbuff = (double *) malloc(sizeof(double));
        double send_buf[] = {result};
        MPI_Reduce((const void *) send_buf, (void *) rbuff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
    }

        MPI_Finalize();
        return 0;}