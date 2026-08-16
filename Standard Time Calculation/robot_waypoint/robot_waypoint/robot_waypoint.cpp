#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

    struct CoordPoint {
        std::string name;
        double x{ 0.0 };
        double y{ 0.0 };
        double z{ 0.0 };
    };

    struct TherbligRow {
        std::string task;
        std::string name;   // R / M / G / RL / END
        std::string from;
        std::string to;
        std::string type;
    };

    struct TaskDefinition {
        std::string task_name;     // 例如 pick and place B
        std::string task_variant;  // 例如 Layout A
        std::vector<TherbligRow> rows;
    };

    struct CsvRow {
        std::vector<std::string> cells;
    };

    std::string trim(const std::string& s) {
        const auto start = s.find_first_not_of(" \t\r\n\"\ufeff");
        if (start == std::string::npos) return "";
        const auto end = s.find_last_not_of(" \t\r\n\"");
        return s.substr(start, end - start + 1);
    }

    std::vector<std::string> split_csv_line(const std::string& line) {
        std::vector<std::string> out;
        std::string cell;
        bool in_quotes = false;

        for (char c : line) {
            if (c == '"') {
                in_quotes = !in_quotes;
            }
            else if (c == ',' && !in_quotes) {
                out.push_back(trim(cell));
                cell.clear();
            }
            else {
                cell.push_back(c);
            }
        }
        out.push_back(trim(cell));
        return out;
    }

    std::vector<CsvRow> read_csv(const std::string& path) {
        std::ifstream fin(path);
        if (!fin.is_open()) {
            throw std::runtime_error("無法開啟 CSV: " + path);
        }

        std::vector<CsvRow> rows;
        std::string line;
        while (std::getline(fin, line)) {
            if (!line.empty() && static_cast<unsigned char>(line.back()) == '\r') {
                line.pop_back();
            }
            if (trim(line).empty()) continue;
            rows.push_back({ split_csv_line(line) });
        }
        return rows;
    }

    std::unordered_map<std::string, CoordPoint> load_coord_map(const std::string& path) {
        auto rows = read_csv(path);
        if (rows.size() < 2) {
            throw std::runtime_error("coord CSV 內容不足");
        }

        std::unordered_map<std::string, CoordPoint> coord_map;
        for (size_t i = 1; i < rows.size(); ++i) {
            const auto& cells = rows[i].cells;
            if (cells.size() < 4) continue;

            const std::string name = trim(cells[0]);
            if (name.empty()) continue;

            CoordPoint p;
            p.name = name;
            p.x = std::stod(cells[1]);
            p.y = std::stod(cells[2]);
            p.z = std::stod(cells[3]);
            coord_map[name] = p;
        }
        return coord_map;
    }

    std::vector<TaskDefinition> load_therblig_tasks(const std::string& path) {
        auto rows = read_csv(path);
        if (rows.size() < 2) {
            throw std::runtime_error("therblig CSV 內容不足");
        }

        std::vector<TaskDefinition> tasks;
        TaskDefinition current;

        for (size_t i = 1; i < rows.size(); ++i) {
            const auto& cells = rows[i].cells;
            TherbligRow row;
            row.task = cells.size() > 0 ? trim(cells[0]) : "";
            row.name = cells.size() > 1 ? trim(cells[1]) : "";
            row.from = cells.size() > 2 ? trim(cells[2]) : "";
            row.to = cells.size() > 3 ? trim(cells[3]) : "";
            row.type = cells.size() > 4 ? trim(cells[4]) : "";

            if (!row.task.empty()) {
                if (row.task.find("pick") != std::string::npos ||
                    row.task.find("Pick") != std::string::npos) {
                    if (!current.task_name.empty()) {
                        tasks.push_back(current);
                    }
                    current = {};
                    current.task_name = row.task;
                }
                else if (row.task.find("Layout") != std::string::npos) {
                    current.task_variant = row.task;
                }
            }

            if (!current.task_name.empty()) {
                current.rows.push_back(row);
                if (row.name == "END") {
                    tasks.push_back(current);
                    current = {};
                }
            }
        }

        if (!current.task_name.empty()) {
            tasks.push_back(current);
        }

        return tasks;
    }

    std::string layout_prefix(const std::string& layout) {
        if (layout == "LA") return "LA";
        if (layout == "LB") return "LB";
        if (layout == "LC") return "LC";
        throw std::runtime_error("未知 layout: " + layout);
    }

    std::string map_symbolic_name(const std::string& symbolic, const std::string& layout) {
        if (symbolic == "AGENT") return "BOT";
        if (symbolic == "OBJA" || symbolic == "OBJA_TOP" ||
            symbolic == "OBJB" || symbolic == "OBJB_TOP" ||
            symbolic == "OBJC" || symbolic == "OBJC_TOP" ||
            symbolic == "OBJD" || symbolic == "OBJD_TOP") {
            return symbolic;
        }

        const std::string prefix = layout_prefix(layout);
        if (symbolic == "T") return prefix + "_T";
        if (symbolic == "T_TOP") return prefix + "_T_TOP";
        if (symbolic == "BL") return prefix + "_BL";
        if (symbolic == "BL_TOP") return prefix + "_BL_TOP";
        if (symbolic == "BR") return prefix + "_BR";
        if (symbolic == "BR_TOP") return prefix + "_BR_TOP";

        throw std::runtime_error("無法映射點位名稱: " + symbolic + " (layout=" + layout + ")");
    }

    CoordPoint adjust_for_stack_height(const CoordPoint& base, int stack_index) {
        static const std::map<int, double> height_map = {
            {1, 9.4}, {2, 8.0}, {3, 6.6}, {4, 5.2}, {5, 3.8}, {6, 2.2} };

        auto it = height_map.find(stack_index);
        if (it == height_map.end()) {
            throw std::runtime_error("無效 stack index: " + std::to_string(stack_index));
        }

        CoordPoint p = base;
        const bool is_top = (p.name.size() >= 4 && p.name.substr(p.name.size() - 4) == "_TOP");
        if (!is_top) {
            p.z = it->second;
        }
        return p;
    }

    bool needs_stack_height_adjust(const std::string& resolved_name) {
        return resolved_name == "OBJA" || resolved_name == "OBJB" ||
            resolved_name == "OBJC" || resolved_name == "OBJD";
    }

    CoordPoint resolve_point(const std::unordered_map<std::string, CoordPoint>& coord_map,
        const std::string& symbolic_name,
        const std::string& layout,
        int stack_index) {
        const std::string resolved = map_symbolic_name(symbolic_name, layout);
        auto it = coord_map.find(resolved);
        if (it == coord_map.end()) {
            throw std::runtime_error("座標表找不到點位: " + resolved);
        }

        CoordPoint point = it->second;
        if (needs_stack_height_adjust(resolved)) {
            point = adjust_for_stack_height(point, stack_index);
        }
        return point;
    }

    std::vector<std::string> layouts_for_task(const std::string&) {
        return { "LA", "LB", "LC" };
    }

    void write_output_csv(const std::string& output_path,
        const std::unordered_map<std::string, CoordPoint>& coord_map,
        const std::vector<TaskDefinition>& tasks) {
        std::ofstream fout(output_path);
        if (!fout.is_open()) {
            throw std::runtime_error("無法建立輸出 CSV: " + output_path);
        }

        fout << "task,variant,layout,stack,action,from,to,resolved_to,x,y,z\n";

        for (const auto& task : tasks) {
            if (task.task_name == "replace") continue;

            const auto layouts = layouts_for_task(task.task_name);
            for (const auto& layout : layouts) {
                for (int stack_index = 1; stack_index <= 6; ++stack_index) {

                    fout << '\n';
                    fout << "# Task: " << task.task_name
                        << " | Variant: " << task.task_variant
                        << " | Layout: " << layout
                        << " | Stack: " << stack_index << '\n';

                    for (const auto& row : task.rows) {
                        if (row.name != "R" && row.name != "M") continue;
                        if (row.to.empty()) continue;

                        CoordPoint point = resolve_point(coord_map, row.to, layout, stack_index);
                        const std::string resolved_to = map_symbolic_name(row.to, layout);

                        fout << task.task_name << ','
                            << task.task_variant << ','
                            << layout << ','
                            << stack_index << ','
                            << row.name << ','
                            << row.from << ','
                            << row.to << ','
                            << resolved_to << ','
                            << point.x / 100.0 << ','
                            << point.y / 100.0 << ','
                            << point.z / 100.0 << '\n';
                    }
                }
            }
        }
    }

}  // namespace

int main() {
    const std::string coord_csv = "coord_for_robot.csv";
    const std::string therblig_csv = "therblig_calculation.csv";
    const std::string output_csv = "robot_task_waypoint.csv";

    try {
        auto coord_map = load_coord_map(coord_csv);
        auto tasks = load_therblig_tasks(therblig_csv);
        write_output_csv(output_csv, coord_map, tasks);

        std::cout << "已完成輸出: " << output_csv << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "程式失敗: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
