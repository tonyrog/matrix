%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    Generate text from values
%%% @end
%%% Created :  3 Oct 2017 by Tony Rogvall <tony@rogvall.se>

-module(matrix_gen_op).

-export([file/0, file/1]).

file() ->
    file(filename:join(code:priv_dir(matrix),"matrix_op.term")).

file(File) ->
    {ok, Sections} = file:consult(File),
    sections(Sections, []).

sections([{text,Text}|Sections], Bound) ->
    _Text1 = expand_text(Text, Bound),
    %% io:put_chars(Text1),
    sections(Sections, Bound);
sections([{subsection,SubSections}|Sections], Bound) ->
    sections(SubSections, Bound),
    sections(Sections, Bound);
sections([{Key,Values}|Sections],Bound) when is_atom(Key) ->
    sections(Sections, [{Key,Values}|Bound]);
sections([{Keys,Values}|Sections],Bound) when is_tuple(Keys) ->
    sections(Sections, [{Keys,Values}|Bound]);
sections([], _Bound) ->
    ok.
    
expand_text(Text, Bound) ->
    TextList = split_text(Text),
    Vars = lists:filter(fun erlang:is_atom/1, TextList),
    Bound1 = lists:filter(fun({A,_}) -> var_is_used(A,Vars) end, Bound),
    io:format("bound = ~w\n", [Bound1]),
    expand_all(Bound1, TextList, []).

expand_all([], Text, Vars) ->
    Expanded = [ if is_atom(T) ->
			 to_string(proplists:get_value(T,Vars));
		    is_integer(T) -> T
		 end || T <- Text ],
    io:format("vars = ~p\n", [Vars]),
    io:put_chars(Expanded);
expand_all([{K,[V]}|Bound], Text, Vars) when is_atom(K) ->
    expand_all(Bound, Text, [{K,V}|Vars]);
expand_all([{K,[V|Vs]}|Bound], Text, Vars) when is_atom(K) ->
    expand_all(Bound, Text, [{K,V}|Vars]),
    expand_all([{K,Vs}|Bound], Text, Vars);
expand_all([{T,[V]}|Bound], Text, Vars) when is_tuple(T) ->
    Vs1 = lists:zip(tuple_to_list(T), tuple_to_list(V)),
    expand_all(Bound, Text, Vs1++Vars);
expand_all([{T,[V|Vs]}|Bound], Text, Vars) when is_tuple(T) ->
    Vs1 = lists:zip(tuple_to_list(T), tuple_to_list(V)),
    expand_all(Bound, Text, Vs1++Vars),
    expand_all([{T,Vs}|Bound], Text, Vars).


to_string(A) when is_atom(A) -> atom_to_list(A);
to_string(A) when is_integer(A) -> integer_to_list(A);
to_string(A) when is_list(A) -> A.


var_is_used(A, Vars) when is_atom(A) ->
    lists:member(A, Vars);
var_is_used(T, Vars) when is_tuple(T) ->
    lists:max([var_is_used(V, Vars) || V <- tuple_to_list(T)]);
var_is_used(_, _Vars) ->
    false.

var_is_bound(A, [{A,_}|_]) -> true;
var_is_bound(A, [{V,_}|Bs]) when is_atom(V) -> var_is_bound(A, Bs);
var_is_bound(A, [{T,_}|Bs]) when is_tuple(T) ->
    case lists:memeber(A, tuple_to_list(T)) of
	true -> true;
	false -> var_is_bound(A, Bs)
    end;
var_is_bound(_A, []) ->
    false.

split_text([$$,${|Cs]) ->
    split_text_var(Cs,[]);
split_text([C|Cs]) ->
    [C|split_text(Cs)];
split_text([]) ->
    [].

split_text_var([$}|Cs],[]) ->
    split_text(Cs);
split_text_var([$}|Cs],Var) ->
    try list_to_existing_atom(lists:reverse(Var)) of
	Atom ->
	    [Atom | split_text(Cs)]
    catch
	error:_ ->
	    split_text(Cs)
    end;
split_text_var([C|Cs],Var) ->
    split_text_var(Cs,[C|Var]);
split_text_var([],_Var) -> %% warning?
    [].



    





    



    


    
