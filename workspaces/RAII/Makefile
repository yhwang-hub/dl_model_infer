cpp_scrs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_scrs))
cpp_objs := $(subst src,objs,$(cpp_objs))

# 定义文件夹名称参数
workspace := build
binary := pro

run : pro 
	@cd $(workspace) && ./$(binary)

# 定义pro快捷编译指令，这里这里只发生编译，不执行
pro : $(workspace)/$(binary)

$(workspace)/$(binary) : $(cpp_objs)
	@mkdir -p $(dir $@)
	@g++ $^ -o $@

objs/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	@g++ -Isrc -c $< -o $@ -g

# 需要调试
clean:
	@rm -rf ./${workspace} ./objs

.PHONY: debug

debug:
	@echo ${cpp_scrs}
	@echo ${cpp_objs}

