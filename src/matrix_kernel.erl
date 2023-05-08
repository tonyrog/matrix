%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2023, Tony Rogvall
%%% @doc
%%%    Matrix kernel utilities
%%% @end
%%% Created : 26 Feb 2023 by Tony Rogvall <tony@rogvall.se>

-module(matrix_kernel).

-export([validate/1]).
-export([validate/2]).

-compile(export_all).

-type aop1() :: neg | inv.
-type bop1() :: 'bnot'.
-type op1()  :: mov | ret | aop1() | bop1().

-type aop2() :: add | sub | mul.
-type cop2() :: cmplt | cmple | cmpeq.
-type bop2() ::  'band' | 'bor' | 'bxor'.
-type op2()  :: aop2() | bop2() | cop2().

-type vaop1() :: vneg | vinv.
-type vbop1() :: vbnot.
-type vop1()  :: vmov | vret | vaop1() | vbop1().

-type vaop2() :: vadd | vsub | vmul.
-type vcop2() :: vcmplt | vcmple | vcmpeq.
-type vbop2() :: vband | vbor | vbxor. 
-type vop2()  :: vaop2() | vbop2() | vcop2().

-type iop1() :: op1() | vop1().
-type iop2() :: op2() | vop2().

-type int_t()    :: int8|int16|int32|int64|int128.
-type uint_t()   :: uint8|uint16|uint32|uint64|uint128.
-type float_t()  :: float16|float32|float64.
-type ityp() :: int_t() | uint_t() | float_t().

-type reg() :: {'r', 0..15}.
-type arg() :: {'a', 0..15}.
-type const() :: {'c',integer()|float()}.
-type sreg() :: reg() | arg() | const().
-type dreg() :: reg().
-type creg() :: reg().

-type inst1() :: {iop1(),ityp(),dreg(),sreg()}.
-type inst2() :: {iop2(),ityp(),dreg(),sreg(),sreg()}.

-type cinst1() :: {op1(),ityp(),dreg(),sreg(),creg()}.
-type cinst2() :: {op2(),ityp(),dreg(),sreg(),sreg(),creg()}.

-type inst() :: inst1() | inst2() | cinst1() | cinst2().

-export_type([inst/0]).

-spec validate(Prog::[inst()]) ->
	  {ok, Prog1::[inst()]} |
	  {error, [{integer(),inst()}]}.

validate(Prog) ->
    validate(Prog, 0).

validate(Prog,Addr) ->
    validate_(Prog, [], Addr, []).


validate_([I|Is], Js, Addr, Err) ->
    case valid_inst(I) of
	false ->
	    io:format("~w: ~p : invalid instruction\n", [Addr, I]),
	    validate_(Is, Js, Addr+1, [{Addr,I}|Err]);
	Inst ->
	    validate_(Is, [Inst|Js], Addr+1, Err)
    end;
validate_([], Js, _Addr, []) ->
    {ok, lists:reverse(Js)};
validate_([], _Js, _Addr, Err) ->
    {error, lists:reverse(Err)}.

valid_inst({I,A}) ->
    valid_inst(valid_i(I), [A]);
valid_inst({I,A,B}) ->
    valid_inst(valid_i(I), [A,B]);
valid_inst({I,A,B,C}) ->
    valid_inst(valid_i(I), [A,B,C]);
valid_inst({I,A,B,C,D}) ->
    valid_inst(valid_i(I), [A,B,C,D]);
valid_inst(_) ->
    false.

valid_inst({ret,false,T},[Ri]) ->
    case valid_reg(Ri) of
	false -> false;
	true -> {{ret,false,T},Ri}
    end;
valid_inst({mov,false,T},[Ri,Rd]) ->
    case valid_const_arg_or_reg(Ri) of
	false -> false;
	true ->
	    case valid_reg(Rd) of
		false -> false;
		true -> {{mov,false,T},Ri,Rd}
	    end
    end;
valid_inst({mov,true,T},[Ri,Rd,Rc]) ->
    case valid_const_arg_or_reg(Ri) of
	false -> false;
	true ->
	    case valid_reg(Rd) and valid_reg(Rc) of
		false -> false;
		true -> {{mov,true,T},Ri,Rd,Rc}
	    end
    end;
valid_inst({Instr,false,Type},[Ri,Rd]) -> 
    case valid_reg(Ri) and valid_reg(Rd) of
	false -> false;
	true -> {{Instr,false,Type},Ri,Rd}
    end;
valid_inst({Instr,false,Type},[Ri,Rj,Rd]) ->
    case valid_reg(Ri) and valid_reg(Rj) and valid_reg(Rd) of
	false -> false;
	true -> {{Instr,false,Type},Ri,Rj,Rd}
    end;
valid_inst({Instr,true,Type},[Ri,Rd,Rc]) -> 
    case valid_reg(Ri) and valid_reg(Rd) of
	false -> false;
	true -> {{Instr,true,Type},Ri,Rd,Rc}
    end;	
valid_inst({Instr,true,Type},[Ri,Rj,Rd,Rc]) ->
    case valid_reg(Ri) and valid_reg(Rj) and valid_reg(Rd) and valid_reg(Rc) of
	false -> false;
	true -> {{Instr,true,Type},Ri,Rj,Rd,Rc}
    end;
valid_inst(_, _) -> false.


valid_i(I) when is_atom(I) ->
    valid_i(atom_to_list(I));
valid_i(I) when is_binary(I) ->
    valid_i(binary_to_list(I));
valid_i(Str) when is_list(Str) ->
    case string:split(string:to_lower(Str), ".", all) of
	[I] -> 
	    case valid_iname(I) of
		false -> false;
		IA -> {IA,false,int128}
	    end;
	[I,"c"] -> 
	    case valid_iname(I) of
		false -> false;
		IA -> {IA,true,int128}
	    end;
	[I,T] -> 
	    case valid_iname(I) of
		false -> false;
		IA ->
		    case valid_tname(T) of
			false -> false;
			TA -> {IA,false,TA}
		    end
	    end;
	[I,"c",T] -> 
	    case valid_iname(I) of
		false -> false;
		IA ->
		    case valid_tname(T) of
			false -> false;
			TA -> {IA,true,TA}
		    end
	    end
    end.

valid_iname("ret")    -> 'ret';
valid_iname("mov")    -> 'mov';
valid_iname("neg")    -> 'neg';
valid_iname("inv")    -> 'inv';
valid_iname("bnot")   -> 'bnot';
valid_iname("add")    -> 'add';
valid_iname("sub")    -> 'sub';
valid_iname("mul")    -> 'mul';
valid_iname("band")   -> 'band';
valid_iname("bor")    -> 'bor';
valid_iname("bxor")   -> 'bxor';
valid_iname("lt")     -> 'lt';
valid_iname("lte")    -> 'lte';
valid_iname("eq")     -> 'eq';
%% vector version
valid_iname("vret")   -> 'vret';
valid_iname("vmov")   -> 'vmov';
valid_iname("vneg")   -> 'vneg';
valid_iname("vinv")   -> 'vinv';
valid_iname("vbnot")  -> 'vbnot';
valid_iname("vadd")   -> 'vadd';
valid_iname("vsub")   -> 'vsub';
valid_iname("vmul")   -> 'vmul';
valid_iname("vband")  -> 'vband';
valid_iname("vbor")   -> 'vbor';
valid_iname("vbxor")  -> 'vbxor';
valid_iname("vcmplt") -> 'vcmplt';
valid_iname("vcmple") -> 'vcmple';
valid_iname("vcmpeq") -> 'vcmpeq';

valid_iname(_) -> false.

valid_tname("u8")  -> uint8;
valid_tname("u16") -> uint16;
valid_tname("u32") -> uint32;
valid_tname("u64") -> uint64;
valid_tname("u128") -> uint128;
valid_tname("i8")  -> int8;
valid_tname("i16") -> int16;
valid_tname("i32") -> int32;
valid_tname("i64") -> int64;
valid_tname("i128") -> int128;
valid_tname("f16") -> float16;
valid_tname("f32") -> float32;
valid_tname("f64") -> float64;
valid_tname(_) -> false.
    
valid_reg({r,I}) when I >= 0, I =< 15 -> true;
valid_reg({v,I}) when I >= 0, I =< 15 -> true;  %% fixme: only for vop!
valid_reg(_) -> false.

valid_arg({a,I}) when I >= 0, I =< 15 -> true;
valid_arg(_) -> false.

valid_const({c,I}) when is_integer(I) -> true;
valid_const({c,F}) when is_float(F) -> true;
valid_const(_) -> false.

valid_arg_or_reg(A) -> valid_arg(A) or valid_reg(A).
valid_const_arg_or_reg(A) -> valid_const(A) or valid_arg(A) or valid_reg(A).
