#ifndef MESH_MATRIX_ROW_READERS
#define MESH_MATRIX_ROW_READERS
// Both product algorithms consume the same logical scalar reader. Layouts
// and computed-index readers are specialized by the shared page lowering.
template<uint Binding,typename Scalar,bool Transposed,uint Columns,bool Affine,
         long RowStride,long ColumnStride,long BatchStride,typename Read>
struct matrix_row_read {
    Read reader;ulong base;
    ulong ordinal(uint row,uint column) const {
        return base+(Transposed?ulong(column)*Columns+row:ulong(row)*Columns+column);
    }
    float scalar(uint row,uint column) const {return float(reader.load(ordinal(row,column)));}
    vec<float,MESH_LINEAR_VECTOR> vector(uint row,uint column) const {
        if constexpr(Affine&&ColumnStride==1&&RowStride%MESH_LINEAR_VECTOR==0&&BatchStride%MESH_LINEAR_VECTOR==0)
            if(mesh_index<Binding>(0)%MESH_LINEAR_VECTOR==0) {
                using Vector=vec<Scalar,MESH_LINEAR_VECTOR>;
                if constexpr(mesh_observed<Binding>())
                    return vec<float,MESH_LINEAR_VECTOR>(mesh_step_read<Vector>((device const uchar *)
                        (reader.values+mesh_index<Binding>(ordinal(row,column))),0));
                else return vec<float,MESH_LINEAR_VECTOR>(*(device const Vector*)(reader.values+mesh_index<Binding>(ordinal(row,column))));
            }
        vec<float,MESH_LINEAR_VECTOR> result;
        for(uint k=0;k<MESH_LINEAR_VECTOR;k++)result[k]=scalar(row,column+k);
        return result;
    }
};
template<uint Binding,typename Output,typename Bias,uint Rows,uint Columns,uint BiasRows,uint BiasColumns,bool Activate> struct matrix_row_store {
    device uchar* values;Bias bias;ulong batch;float alpha,beta;
    void operator()(uint row,uint column,float total) const {
        float value=alpha*total;
        if constexpr(BiasColumns>0)value+=beta*float(bias.load((BiasRows>1?row:0)*BiasColumns+(BiasColumns>1?column:0)));
        mesh_step_write<Output>(values,mesh_index<Binding>((batch*Rows+row)*Columns+column),
            Output(Activate?tanh(value):value));
    }
};
#endif
