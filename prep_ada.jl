using ArgParse
using AdaGram
using JSON
using NPZ

function main(args)

    s = ArgParseSettings(description = "Prepare word vectors for Input-Adaptive conditioning")

    @add_arg_table s begin
        "--defs"
            nargs = '+'
            arg_type = String
            required = true
            help = "location of json file with definitions."
        "--save"
            nargs = '+'
            arg_type = String
            required = true
            help = "where to save files"
        "--ada" 
            arg_type = String
            required = true
            help = "location of AdaGram file"
    end

    parsed_args = parse_args(s)
    if length(parsed_args["defs"]) != length(parsed_args["save"])
        error("Number of defs files must match number of save locations")
    end

    vm, dict = load_model(parsed_args["ada"]);
    for i = 1:length(parsed_args["defs"])
        open(parsed_args["defs"][i], "r") do f
            global definitions = JSON.parse(readstring(f))
        end
        global vectors = zeros(length(definitions), length(vm.In[:, 1, 1]))
        for (k, elem) in enumerate(definitions)
            if haskey(dict.word2id, elem[1][1])
                global good_context = []
                for w in elem[3]
                    if haskey(dict.word2id, w)
                        push!(good_context, w)
                    end
                end
                mxval, mxidx = findmax(disambiguate(vm, dict, elem[1][1], split(join(good_context, " "))))
                vectors[k, :] = vm.In[:, mxidx, dict.word2id[elem[1][1]]]
            end
        end
        npzwrite(parsed_args["save"][i], vectors)
    end

end

main(ARGS)